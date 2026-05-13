#!/usr/bin/env python3
"""MSA formalize() pipeline (Wong et al. 2025, arXiv:2507.12547).

Three steps run for a single target test scenario:

  1. Parse the natural-language conditions and queries into WebPPL stubs.
  2. Generate K candidate informal-knowledge + dependency-graph descriptions
     in parallel, LLM-score each, keep the best.
  3. Generate the full WebPPL model from scenario + parse + best graph.

In-context demos are the four train scenarios that don't match the test
sport (e.g. testing a biathalon vignette uses canoe-race + tug-of-war +
diving + exam as examples).

Public API:
  load_train_scenarios()          -> {name: {field: str}}
  load_test_scenario(scn, expt)   -> {background, conditions, query}
  formalize(target, exclude_sport, llm, ...) -> (model_src, parse_block)

The LLM call goes through litellm with OpenRouter; .env supplies the key.
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
TRAIN_DIR = REPO / 'data' / 'train-scenarios'
TEST_DIR = REPO / 'data' / 'test-scenarios'
PROMPT_DIR = REPO / 'prompts'

load_dotenv(REPO / '.env')

# Short label -> OpenRouter model id. Override via env var MSA_LLM_PRO etc.
LLM_MAP = {
    'pro':    os.environ.get('MSA_LLM_PRO',
                             'openrouter/google/gemini-3.1-pro-preview'),
    'flash':  os.environ.get('MSA_LLM_FLASH',
                             'openrouter/google/gemini-3-flash-preview'),
    'claude': os.environ.get('MSA_LLM_CLAUDE',
                             'openrouter/anthropic/claude-opus-4-6'),
    'sonnet': os.environ.get('MSA_LLM_SONNET',
                             'openrouter/anthropic/claude-sonnet-4-6'),
    'gpt':    os.environ.get('MSA_LLM_GPT', 'openrouter/openai/gpt-5'),
}


# --------------------------------------------------------------------------- #
# Scenario loading + example formatting
# --------------------------------------------------------------------------- #

def _join(field):
    return '\n'.join(field) if isinstance(field, list) else field


def load_train_scenarios():
    """Return {name: {field: str}} for every train scenario."""
    out = {}
    for jp in sorted(TRAIN_DIR.glob('*.json')):
        data = json.loads(jp.read_text())
        out[jp.stem] = {k: _join(v) for k, v in data.items()}
    return out


def _split_test_text(text):
    """Split a plain-text test scenario into background/conditions/query."""
    lines = text.splitlines()
    heads = ['BACKGROUND', 'CONDITIONS', 'QUERIES']
    out_keys = ['background', 'conditions', 'query']
    idxs = []
    for h in heads:
        for i, ln in enumerate(lines):
            if ln.strip() == h:
                idxs.append(i)
                break
        else:
            raise ValueError(f"Missing heading {h!r} in test scenario")
    idxs.append(len(lines))
    out = {}
    for key, s, e in zip(out_keys, idxs, idxs[1:]):
        section = lines[s + 1:e]
        while section and not section[0].strip():
            section.pop(0)
        while section and not section[-1].strip():
            section.pop()
        out[key] = '\n'.join(section)
    return out


def load_test_scenario(scenario_id, expt):
    path = TEST_DIR / f'e{expt}' / 'scenarios' / f'{scenario_id}.txt'
    if not path.exists():
        raise SystemExit(f"Test scenario not found: {path}")
    return _split_test_text(path.read_text())


def extract_sport(scenario_id):
    """Pull the sport prefix off a test scenario id."""
    for sport in ('biathalon', 'canoe-race', 'tug-of-war'):
        if scenario_id.startswith(sport + '_'):
            return sport
    raise ValueError(f"Cannot extract sport from: {scenario_id}")


# Block formatters --------------------------------------------------------- #

def _scenario_block(s):
    return ('<START_SCENARIO>\n'
            f'BACKGROUND\n{s["background"]}\n\n'
            f'CONDITIONS\n{s["conditions"]}\n\n'
            f'QUERIES\n{s["query"]}\n'
            '<END_SCENARIO>')


def _parse_block(s):
    return ('<START_LANGUAGE_TO_WEBPPL_CODE>\n'
            f'// CONDITIONS\n{s["parsed_conditions"]}\n\n'
            f'// QUERIES\n{s["parsed_queries"]}\n'
            '<END_LANGUAGE_TO_WEBPPL_CODE>')


def _scratchpad_block(s):
    return ('<START_SCRATCHPAD>\n'
            f'{s["informal"]}\n\n'
            f'<START_CONCEPT_TRACE>\n{s["graph"]}\n<END_CONCEPT_TRACE>\n'
            '<END_SCRATCHPAD>')


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
    m = re.search(r'\b(10|[0-9])\b', text)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------- #
# LLM call (litellm via OpenRouter)
# --------------------------------------------------------------------------- #

def generate(prompt, llm, *, temperature=0.2, system=None, max_tokens=16384):
    import litellm
    litellm.suppress_debug_info = True
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    model_id = LLM_MAP.get(llm, llm)
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})

    resp = litellm.completion(
        model=model_id, messages=messages,
        temperature=temperature, max_tokens=max_tokens, num_retries=3,
    )
    return resp.choices[0].message.content or ''


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def _shuffled(items, rng):
    out = list(items)
    rng.shuffle(out)
    return out


def formalize(target, exclude_sport, llm='flash', *,
              k_graph=4, save_intermediates=True,
              intermediates_dir=None, rng=None, verbose=False):
    """Run the 3-step pipeline.

    target: dict with 'background', 'conditions', 'query' (the test scenario).
    exclude_sport: train-scenario name to omit from in-context demos
                   (must match a key in data/train-scenarios/).

    Returns (model_src, parse_block) — the final WebPPL model body (markers
    stripped) and the step-1 <START_LANGUAGE_TO_WEBPPL_CODE>...<END_…>
    block (markers kept). The parse block is returned separately so callers
    can re-use the parsed conditions/queries to build a gold model.
    """
    if rng is None:
        rng = random.Random(0)

    train = load_train_scenarios()
    if exclude_sport not in train:
        raise SystemExit(
            f"Unknown exclude_sport: {exclude_sport!r}. "
            f"Train scenarios: {sorted(train)}")
    others = [s for n, s in train.items() if n != exclude_sport]

    system_prompt = (PROMPT_DIR / 'generate-system-prompt.txt').read_text()

    if save_intermediates:
        if intermediates_dir is None:
            raise ValueError(
                "save_intermediates=True requires intermediates_dir")
        intermediates_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg)

    def save(name, content):
        if save_intermediates:
            (intermediates_dir / name).write_text(content)

    # ---------------- Step 1: parse ------------------------------------- #
    log("[1/3] parse")
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

    # ---------------- Step 2: graph (K parallel) ------------------------ #
    log(f"[2/3] {k_graph} graphs")
    score_template = (PROMPT_DIR / 'score-graph.txt').read_text()
    target_after_parse = _scenario_block(target) + '\n\n' + parse_block

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
            else:
                save(f'2_graph_{k}_response.txt', gresp)
            log(f"  graph {k+1}/{k_graph}: score={score}")

    if not any(g for g in graphs):
        raise SystemExit("All graph generations failed")

    best_idx = max(range(len(graphs)), key=lambda i: scores[i])
    best_graph = graphs[best_idx]
    save('2_graph_best.txt',
         f'# Best graph #{best_idx} (score={scores[best_idx]})\n\n{best_graph}')

    # Score summary table
    summary = [f"# {k_graph} graph candidates (best: #{best_idx})", "",
               f"{'k':>3}  {'score':>5}  {'chars':>7}"]
    for k, (g, sc) in enumerate(zip(graphs, scores)):
        mk = ' *' if k == best_idx else '  '
        summary.append(f"{k:>3}  {sc:>5}  {len(g):>7}{mk}")
    save('2_graph_scores.txt', '\n'.join(summary) + '\n')

    # ---------------- Step 3: model ------------------------------------- #
    log("[3/3] model")
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

    return model_src, parse_block
