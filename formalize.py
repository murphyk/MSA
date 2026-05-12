#!/usr/bin/env python3
"""MSA formalize() pipeline.

Three-step procedure adapted from Wong et al. 2025 (arXiv:2507.12547):

  1. Parse the natural-language conditions and queries into WebPPL stubs.
  2. Generate K candidate informal-knowledge + dependency-graph descriptions,
     LLM-score each, keep the best.
  3. Generate the full WebPPL model from scenario + parse + best graph.

Each step is in-context prompted with the other 4 scenarios as examples.

The LLM SDK (google-genai / anthropic / openai) is imported lazily inside
`generate()` so that other tooling in the repo doesn't require it.
"""

from __future__ import annotations

import json
import os
import random
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent
SCENARIO_DIR = REPO / 'scenarios'
PROMPT_DIR = REPO / 'prompts'

# Load OPENROUTER_API_KEY etc. from .env
load_dotenv(REPO / '.env')

# Short label -> OpenRouter model id. Override via env var MSA_LLM_PRO etc.
# Pass any unrecognised label through verbatim as the model id, so callers
# can spell out e.g. --llm openrouter/x-ai/grok-4 directly.
LLM_MAP = {
    'pro':    os.environ.get('MSA_LLM_PRO',
                             'openrouter/google/gemini-3.1-pro-preview'),
    'flash':  os.environ.get('MSA_LLM_FLASH',
                             'openrouter/google/gemini-3-flash-preview'),
    'claude': os.environ.get('MSA_LLM_CLAUDE',
                             'openrouter/anthropic/claude-opus-4-6'),
    'sonnet': os.environ.get('MSA_LLM_SONNET',
                             'openrouter/anthropic/claude-sonnet-4-6'),
    'gpt':    os.environ.get('MSA_LLM_GPT',
                             'openrouter/openai/gpt-5'),
}

DEFAULT_K_GRAPH = 8


# --------------------------------------------------------------------------- #
# Scenario loading + example formatting
# --------------------------------------------------------------------------- #

def _join(field):
    return '\n'.join(field) if isinstance(field, list) else field


def load_all_scenarios():
    """Return {scenario_name: {field: str}} with line-list fields joined."""
    out = {}
    for jp in sorted(SCENARIO_DIR.glob('*.json')):
        data = json.loads(jp.read_text())
        out[jp.stem] = {k: _join(v) for k, v in data.items()}
    return out


def _scenario_block(s):
    return (
        '<START_SCENARIO>\n'
        f'BACKGROUND\n{s["background"]}\n\n'
        f'CONDITIONS\n{s["conditions"]}\n\n'
        f'QUERIES\n{s["query"]}\n'
        '<END_SCENARIO>'
    )


def _parse_block(s):
    return (
        '<START_LANGUAGE_TO_WEBPPL_CODE>\n'
        f'// CONDITIONS\n{s["parsed_conditions"]}\n\n'
        f'// QUERIES\n{s["parsed_queries"]}\n'
        '<END_LANGUAGE_TO_WEBPPL_CODE>'
    )


def _scratchpad_block(s):
    return (
        '<START_SCRATCHPAD>\n'
        f'{s["informal"]}\n\n'
        f'<START_CONCEPT_TRACE>\n{s["graph"]}\n<END_CONCEPT_TRACE>\n'
        '<END_SCRATCHPAD>'
    )


def _model_block(s):
    return f'<START_WEBPPL_MODEL>\n{s["model"]}\n<END_WEBPPL_MODEL>'


def _example_for_parse(s):
    return _scenario_block(s) + '\n\n' + _parse_block(s)


def _example_for_graph(s):
    return _example_for_parse(s) + '\n\n' + _scratchpad_block(s)


def _example_for_model(s):
    return _example_for_graph(s) + '\n\n' + _model_block(s)


# --------------------------------------------------------------------------- #
# Prompt-template filling and response extraction
# --------------------------------------------------------------------------- #

def _fill(template_name, mapping):
    text = (PROMPT_DIR / template_name).read_text()
    for placeholder, value in mapping.items():
        text = text.replace(placeholder, value)
    return text


def _extract_between(text, start_marker, end_marker, *, inclusive=True):
    s = text.find(start_marker)
    if s < 0:
        raise ValueError(
            f"Marker {start_marker!r} not found in LLM response:\n"
            + text[:1000])
    e = text.find(end_marker, s + len(start_marker))
    if e < 0:
        raise ValueError(
            f"Marker {end_marker!r} not found after {start_marker!r}:\n"
            + text[s:s + 1000])
    if inclusive:
        return text[s:e + len(end_marker)]
    return text[s + len(start_marker):e]


def _extract_score(text):
    """Pull the first 0..10 integer from a scoring response. Default 0."""
    m = re.search(r'\b(10|[0-9])\b', text)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------- #
# LLM call (lazy import; provider chosen by model-id prefix)
# --------------------------------------------------------------------------- #

def generate(prompt, llm, *, temperature=0.2, system=None, max_tokens=16384):
    """Provider-agnostic completion via litellm.

    The model id is taken from LLM_MAP[llm] if `llm` is a short label,
    otherwise `llm` is passed through directly. Any provider that litellm
    supports works (see https://docs.litellm.ai/docs/providers).
    """
    import litellm
    litellm.suppress_debug_info = True
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    model_id = LLM_MAP.get(llm, llm)

    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})

    resp = litellm.completion(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=3,
    )
    return resp.choices[0].message.content or ''


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def _shuffled(items, rng):
    out = list(items)
    rng.shuffle(out)
    return out


def formalize(task, llm='pro', expt='1', *,
              k_graph=DEFAULT_K_GRAPH,
              save_intermediates=True,
              intermediates_dir=None,
              rng=None,
              verbose=True):
    """Run the 3-step pipeline. Returns the final WebPPL model source string."""
    if rng is None:
        seed = int(expt) if (expt and str(expt).isdigit()) else 0
        rng = random.Random(seed)

    all_scenarios = load_all_scenarios()
    if task not in all_scenarios:
        raise SystemExit(f"Scenario not found: {task}. "
                         f"Available: {sorted(all_scenarios)}")
    target = all_scenarios[task]
    others = [s for n, s in all_scenarios.items() if n != task]

    system_prompt = (PROMPT_DIR / 'generate-system-prompt.txt').read_text()

    if not expt:
        suffix = ''
    elif str(expt).isdigit():
        suffix = f'_e{expt}'
    else:
        suffix = f'_{expt}'
    method = f'msa_{llm}{suffix}'
    if save_intermediates and intermediates_dir is None:
        intermediates_dir = REPO / 'intermediates' / method / task
    if save_intermediates:
        intermediates_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg)

    def save(name, content):
        if save_intermediates:
            (intermediates_dir / name).write_text(content)

    # ---------------- Step 1: parse ------------------------------------- #
    log("[1/3] Parsing observations and queries...")
    examples = '\n\n'.join(_example_for_parse(s) for s in _shuffled(others, rng))
    prompt = _fill('generate-parsing.txt', {
        '<SHUFFLED EXAMPLES OF SCENARIOS AND START_LANGUAGE_TO_WEBPPL_CODE DELIMITED BLOCK INJECTED HERE>':
            examples,
        '<SCENARIO_INJECTED_HERE>': _scenario_block(target),
    })
    save('1_parse_prompt.txt', prompt)
    resp = generate(prompt, llm, temperature=0.2, system=system_prompt)
    save('1_parse_response.txt', resp)
    parse_block = _extract_between(
        resp, '<START_LANGUAGE_TO_WEBPPL_CODE>', '<END_LANGUAGE_TO_WEBPPL_CODE>')
    save('1_parse_block.txt', parse_block)

    # ---------------- Step 2: causal graph (K samples + scoring) -------- #
    log(f"[2/3] Generating {k_graph} candidate dependency graphs in parallel...")
    score_template = (PROMPT_DIR / 'score-graph.txt').read_text()
    target_after_parse = _scenario_block(target) + '\n\n' + parse_block

    # Pre-shuffle example orderings on the main thread so the RNG stays
    # deterministic regardless of worker completion order.
    example_blocks = [
        '\n\n'.join(_example_for_graph(s) for s in _shuffled(others, rng))
        for _ in range(k_graph)
    ]

    def _one_graph(k):
        prompt = _fill('generate-graph.txt', {
            '<SHUFFLED EXAMPLES UP TO DEPENDENCY GRAPH INJECTED HERE>':
                example_blocks[k],
            '<SCENARIO_AND_PARSE_INJECTED_HERE>': target_after_parse,
        })
        gresp = generate(prompt, llm, temperature=0.5, system=system_prompt)
        try:
            gblock = _extract_between(
                gresp, '<START_SCRATCHPAD>', '<END_SCRATCHPAD>')
        except ValueError:
            return k, '', -1, gresp, ''
        sprompt = score_template.replace(
            '<INJECTED_INFORMAL_BACKGROUND_AND_DEPENDENCY_GRAPH_HERE>', gblock)
        sresp = generate(sprompt, llm, temperature=0.0)
        return k, gblock, _extract_score(sresp), gresp, sresp

    graphs = [''] * k_graph
    scores = [-1] * k_graph
    with ThreadPoolExecutor(max_workers=k_graph) as pool:
        for k, gblock, score, gresp, sresp in pool.map(_one_graph, range(k_graph)):
            graphs[k] = gblock
            scores[k] = score
            if gblock:
                save(f'2_graph_{k}.txt',
                     f'# Score: {score}\n\n{gblock}\n\n# Score response:\n{sresp}')
                log(f"  [graph {k+1}/{k_graph}] score={score}")
            else:
                save(f'2_graph_{k}_response.txt', gresp)
                log(f"  [graph {k+1}/{k_graph}] skipped (no markers)")

    if not any(g for g in graphs):
        raise SystemExit("All graph generations failed (no <START_SCRATCHPAD>)")

    best_idx = max(range(len(graphs)), key=lambda i: scores[i])
    best_graph = graphs[best_idx]
    log(f"  -> chose graph {best_idx} (score={scores[best_idx]})")
    save('2_graph_best.txt',
         f'# Best graph #{best_idx} (score={scores[best_idx]})\n\n{best_graph}')

    # ---------------- Step 3: full model -------------------------------- #
    log("[3/3] Generating full WebPPL model...")
    examples = '\n\n'.join(_example_for_model(s) for s in _shuffled(others, rng))
    target_full = target_after_parse + '\n\n' + best_graph
    prompt = _fill('generate-model.txt', {
        '<SHUFFLED EXAMPLES INCLUDING ALL SCENARIOS, PARSES, INFORMAL KNOWLEDGE, AND DEPENDENCY GRAPH, AND PROBABILISTIC PROGRAMS INJECTED HERE>':
            examples,
        '<SCENARIO, PARSE, INFORMAL KNOWLEDGE, AND DEPENDENCY GRAPH INJECTED HERE>':
            target_full,
    })
    save('3_model_prompt.txt', prompt)
    resp = generate(prompt, llm, temperature=0.2, system=system_prompt)
    save('3_model_response.txt', resp)
    model_block = _extract_between(
        resp, '<START_WEBPPL_MODEL>', '<END_WEBPPL_MODEL>', inclusive=False)
    model_src = model_block.strip()
    save('3_model.wppl', model_src)

    log(f"Done. Intermediates in {intermediates_dir.relative_to(REPO)}")
    return model_src
