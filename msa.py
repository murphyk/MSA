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
import random
import re
import subprocess
import sys
from pathlib import Path

from formalize import formalize as run_formalize, load_all_scenarios


REPO = Path(__file__).resolve().parent


def formalize(scenario_name: str, expt: str, llm: str, k_graph: int = 8,
              rng=None, intermediates_dir=None) -> str:
    """Run the 3-step LLM pipeline to formalize a scenario into WebPPL."""
    return run_formalize(scenario_name, llm=llm, expt=expt, k_graph=k_graph,
                         rng=rng, intermediates_dir=intermediates_dir)


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
    ap.add_argument('--n-runs', type=int, default=1,
                    help='number of independent pipeline runs (default: 1). '
                         'When >1, results land in {method}_r0, _r1, ...; '
                         'no per-run comparison is run (use plot_vs_gold.py).')
    args = ap.parse_args()

    # `_e{n}` for numeric expts (e1, e2, ...) so the digit reads as a label;
    # `_{name}` for word expts (smoke, ablate, ...) to avoid `_esmoke` etc.
    if not args.expt:
        suffix = ''
    elif args.expt.isdigit():
        suffix = f'_e{args.expt}'
    else:
        suffix = f'_{args.expt}'
    base_method = f'msa_{args.llm}{suffix}'

    for run_idx in range(args.n_runs):
        run_method = base_method if args.n_runs == 1 else f'{base_method}_r{run_idx}'

        # Different seed per run so the example-shuffle is independent
        # across runs (the LLM temperature provides the rest of the
        # variation). Single-run mode keeps the default rng for backwards
        # compatibility.
        rng = None
        if args.n_runs > 1:
            seed_base = int(args.expt) if args.expt.isdigit() else \
                        (hash(args.expt) & 0xFFFF)
            rng = random.Random(seed_base * 1000 + run_idx)

        intermediates_dir = REPO / 'intermediates' / run_method / args.scenario

        if args.n_runs > 1:
            print(f"\n=== Run {run_idx + 1}/{args.n_runs} ({run_method}) ===")
        model_src = formalize(args.scenario, args.expt, args.llm, args.k_graph,
                              rng=rng, intermediates_dir=intermediates_dir)
        model_src = wrap_with_query_labels(model_src, args.scenario)
        model_dir = REPO / 'models' / run_method
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'{args.scenario}.wppl'
        model_path.write_text(model_src)
        print(f"Wrote {model_path.relative_to(REPO)}")

        run(['node', str(REPO / 'get_samples.js'),
             '--model', args.scenario,
             '--method', run_method])

    # Per-run comparison vs gold1 only in single-run mode; for n_runs>1,
    # use plot_vs_gold.py to aggregate posterior means across runs.
    if args.n_runs == 1:
        run([sys.executable, str(REPO / 'compare_samples.py'),
             '--model', args.scenario,
             '--method1', 'gold1',
             '--method2', base_method])
    else:
        print(f"\nDone {args.n_runs} runs. To aggregate vs gold:")
        print(f"  python plot_vs_gold.py --method {base_method} "
              f"--scenario {args.scenario}")


if __name__ == '__main__':
    main()
