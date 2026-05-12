#!/usr/bin/env python3
"""MSA pipeline.

For a given scenario, formalize it into a WebPPL model, run inference, then
compare the resulting samples against a gold reference. The experiment label
is folded into the method name (`msa_{llm}_e{expt}`) so the model file,
samples, and comparison directory each carry a single suffix-free model name.

Usage:
    python msa.py --scenario tug [--expt 1] [--llm pro]

Effects (with defaults --expt 1 --llm pro):
    models/msa_pro_e1/{scenario}.wppl       (the formalized model)
    samples/msa_pro_e1/{scenario}.json      (1000 samples + queries)
    comparisons/gold1_vs_msa_pro_e1/{scenario}/   (figures + metrics)
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from formalize import formalize as run_formalize, load_all_scenarios


REPO = Path(__file__).resolve().parent


def formalize(scenario_name: str, expt: str, llm: str, k_graph: int = 8) -> str:
    """Run the 3-step LLM pipeline to formalize a scenario into WebPPL."""
    return run_formalize(scenario_name, llm=llm, expt=expt, k_graph=k_graph)


def build_query_labels(scenario_query):
    """Parse 'Query N: ...' lines from the scenario's query field
    into a dict {queryN: 'short label'}."""
    if isinstance(scenario_query, list):
        lines = scenario_query
    else:
        lines = scenario_query.splitlines()
    labels = {}
    for line in lines:
        m = re.match(r'\s*Query\s+(\d+)\s*:\s*(.+)', line)
        if m:
            labels[f'query{m.group(1)}'] = m.group(2).strip()
    return labels


def wrap_with_query_labels(model_src, scenario_name):
    """Prepend `var queryLabels = {...}` so get_samples.wppl can label queries.

    The LLM produces only the model function (matching the in-context
    examples), so the wrapper supplies the labels separately.
    """
    scenarios = load_all_scenarios()
    labels = build_query_labels(scenarios[scenario_name]['query'])
    if not labels:
        return model_src
    items = ',\n  '.join(f'{k}: {json.dumps(v)}' for k, v in labels.items())
    return f'var queryLabels = {{\n  {items}\n}};\n\n{model_src}'


def run(cmd):
    print('$ ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True,
                    help='scenario name (e.g. tug)')
    ap.add_argument('--expt', default='1',
                    help='experiment label (default: 1)')
    ap.add_argument('--llm', default='flash',
                    help='llm label (default: flash; use pro for Gemini 3.1 Pro)')
    ap.add_argument('--k-graph', type=int, default=8,
                    help='number of candidate dependency graphs to sample (default: 8)')
    args = ap.parse_args()

    # `_e{n}` for numeric expts (e1, e2, ...) so the digit reads as a label;
    # `_{name}` for word expts (smoke, ablate, ...) to avoid `_esmoke` etc.
    if not args.expt:
        suffix = ''
    elif args.expt.isdigit():
        suffix = f'_e{args.expt}'
    else:
        suffix = f'_{args.expt}'
    method = f'msa_{args.llm}{suffix}'

    # 1. Formalize and save model.
    model_src = formalize(args.scenario, args.expt, args.llm, args.k_graph)
    model_src = wrap_with_query_labels(model_src, args.scenario)
    model_dir = REPO / 'models' / method
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'{args.scenario}.wppl'
    model_path.write_text(model_src)
    print(f"Wrote {model_path.relative_to(REPO)}")

    # 2. Run inference. get_samples.js defaults --model-dir to models/{method}.
    run(['node', str(REPO / 'get_samples.js'),
         '--model', args.scenario,
         '--method', method])

    # 3. Compare against gold1.
    run([sys.executable, str(REPO / 'compare_samples.py'),
         '--model', args.scenario,
         '--method1', 'gold1',
         '--method2', method])


if __name__ == '__main__':
    main()
