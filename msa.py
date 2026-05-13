#!/usr/bin/env python3
"""MSA pipeline driver.

For a single test scenario:
    python msa.py --scenario {scenario_id} --expt {1|2} [--llm flash]
                  [--n-runs 3] [--k-graph 4]

For all T=7 test scenarios in a sport:
    python msa.py --sport {biathalon|canoe-race|tug-of-war} --expt {1|2}

Reads each test vignette from data/test-scenarios/e{expt}/scenarios/{s}.txt,
runs the 3-step formalize pipeline N independent times, also synthesises a
'gold-for-this-test-scenario' model from models/gold/{sport}.wppl plus the
LLM-parsed conditions/queries, and runs inference on both. Outputs land at:

    samples/msa_{llm}_e{expt}_r{r}/{scenario}.json   (one per run)
    samples/gold_e{expt}/{scenario}.json             (one per scenario)
    models/msa_{llm}_e{expt}_r{r}/{scenario}.wppl
    models/gold_e{expt}/{scenario}.wppl
    intermediates/msa_{llm}_e{expt}_r{r}/{scenario}/

T x N runs are dispatched in parallel via a ThreadPoolExecutor.
"""

import argparse
import concurrent.futures as cf
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from formalize import (REPO, extract_sport, formalize as run_formalize,
                       load_test_scenario)
from gold_for_test import build_gold_model


# Map: test-data sport name -> train-scenario filename stem (now identical
# after the rename, but kept explicit so changes downstream are localised).
SPORT_TO_TRAIN = {
    'biathalon':  'biathalon',
    'canoe-race': 'canoe-race',
    'tug-of-war': 'tug-of-war',
}


def _suffix(expt):
    if not expt:
        return ''
    return f'_e{expt}' if str(expt).isdigit() else f'_{expt}'


def _seed(expt, run_idx):
    base = int(expt) if str(expt).isdigit() else (hash(expt) & 0xFFFF)
    return base * 1000 + run_idx


def _run(cmd):
    subprocess.run(cmd, check=True)


def _save_msa_run(target, sport, scenario_id, llm, expt, run_idx,
                  k_graph, force):
    """One MSA run: formalize -> save model -> inference. Returns
    (parse_block, cached: bool). If `samples/.../{scenario}.json` and the
    cached parse_block both exist (and force=False), skip the LLM and
    inference entirely and just read the cached parse_block."""
    method = f'msa_{llm}{_suffix(expt)}_r{run_idx}'
    intermediates_dir = REPO / 'intermediates' / method / scenario_id
    samples_path = REPO / 'samples' / method / f'{scenario_id}.json'
    parse_path = intermediates_dir / '1_parse_block.txt'

    if not force and samples_path.exists() and parse_path.exists():
        return parse_path.read_text(), True

    rng = random.Random(_seed(expt, run_idx))
    train_sport = SPORT_TO_TRAIN[sport]
    model_src, parse_block = run_formalize(
        target, exclude_sport=train_sport, llm=llm,
        k_graph=k_graph, intermediates_dir=intermediates_dir, rng=rng)

    labels = _build_query_labels(target['query'])
    model_src = _prepend_query_labels(model_src, labels)
    model_dir = REPO / 'models' / method
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f'{scenario_id}.wppl').write_text(model_src)

    _run(['node', str(REPO / 'get_samples.js'),
          '--model', scenario_id, '--method', method])
    return parse_block, False


def _build_query_labels(scenario_query_text):
    labels = {}
    for ln in scenario_query_text.splitlines():
        m = re.match(r'\s*Query\s+(\d+)\s*:\s*(.+)', ln)
        if m:
            labels[f'query{m.group(1)}'] = m.group(2).strip()
    return labels


def _prepend_query_labels(model_src, labels):
    if not labels:
        return model_src
    items = ',\n  '.join(f'{k}: {json.dumps(v)}' for k, v in labels.items())
    return f'var queryLabels = {{\n  {items}\n}};\n\n{model_src}'


def _save_gold_for_test(target, sport, scenario_id, parse_block, expt, force):
    """Synthesise + run inference on the gold-for-this-test-scenario model.
    Returns True if cached, False if recomputed."""
    method = f'gold{_suffix(expt)}'
    samples_path = REPO / 'samples' / method / f'{scenario_id}.json'

    if not force and samples_path.exists():
        return True

    gold_src = build_gold_model(sport, parse_block, target['query'])
    model_dir = REPO / 'models' / method
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f'{scenario_id}.wppl').write_text(gold_src)

    _run(['node', str(REPO / 'get_samples.js'),
          '--model', scenario_id, '--method', method])
    return False


def _process_one_scenario(scenario_id, args):
    """Run N MSA pipelines + 1 gold pipeline for a single test scenario.
    Honours args.force; outputs that already exist on disk are skipped
    when force=False (the default).

    Returns (scenario_id, ok: bool, message: str).
    """
    try:
        sport = extract_sport(scenario_id)
        target = load_test_scenario(scenario_id, args.expt)

        parse_blocks = [None] * args.n_runs
        cached_runs = [False] * args.n_runs
        with cf.ThreadPoolExecutor(max_workers=args.n_runs) as pool:
            futures = {
                pool.submit(_save_msa_run, target, sport, scenario_id,
                            args.llm, args.expt, r, args.k_graph,
                            args.force): r
                for r in range(args.n_runs)
            }
            for fut in cf.as_completed(futures):
                r = futures[fut]
                parse_blocks[r], cached_runs[r] = fut.result()

        gold_cached = _save_gold_for_test(
            target, sport, scenario_id, parse_blocks[0], args.expt, args.force)

        n_cached = sum(cached_runs)
        n_fresh = args.n_runs - n_cached
        bits = []
        if n_fresh:
            bits.append(f'{n_fresh} MSA fresh')
        if n_cached:
            bits.append(f'{n_cached} MSA cached')
        bits.append('gold cached' if gold_cached else 'gold fresh')
        return scenario_id, True, 'ok (' + ', '.join(bits) + ')'
    except Exception as e:
        return scenario_id, False, f"FAILED: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--scenario', help='single test scenario id (no .txt)')
    g.add_argument('--sport',
                   choices=list(SPORT_TO_TRAIN),
                   help='sweep all T=7 test scenarios for this sport')
    ap.add_argument('--expt', default='1',
                    help='experiment number 1 or 2 (default: 1)')
    ap.add_argument('--llm', default='flash',
                    help='llm label (default: flash)')
    ap.add_argument('--n-runs', type=int, default=3,
                    help='simulated participants per scenario (default: 3)')
    ap.add_argument('--k-graph', type=int, default=4,
                    help='candidate dependency graphs (default: 4)')
    ap.add_argument('--max-parallel', type=int, default=4,
                    help='max scenarios in flight at once (default: 4)')
    ap.add_argument('--force', action='store_true',
                    help='ignore existing samples/intermediates and '
                         'recompute everything (default: skip outputs '
                         'that already exist on disk)')
    args = ap.parse_args()

    if args.scenario:
        scenarios = [args.scenario]
    else:
        test_ids = json.loads(
            (REPO / 'data' / 'test_ids.json').read_text())
        scenarios = [f'{args.sport}_{sfx}' for sfx in test_ids[args.sport]]

    print(f"Processing {len(scenarios)} scenario(s) "
          f"with N={args.n_runs} runs, K={args.k_graph} graphs, "
          f"llm={args.llm}, expt={args.expt}")

    t0 = time.time()
    results = []
    workers = min(args.max_parallel, len(scenarios))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one_scenario, s, args): s
                   for s in scenarios}
        for fut in cf.as_completed(futures):
            sid, ok, msg = fut.result()
            results.append((sid, ok, msg))
            tag = '✓' if ok else '✗'
            print(f"  {tag} {sid}: {msg}")

    elapsed = time.time() - t0
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\nDone in {elapsed:.1f}s: {n_ok}/{len(results)} ok")

    if args.sport and n_ok > 0:
        method = f'msa_{args.llm}{_suffix(args.expt)}'
        print(f"\nTo plot vs gold for this sport:")
        print(f"  python plot_vs_gold.py --sport {args.sport} "
              f"--method {method} --expt {args.expt}")


if __name__ == '__main__':
    main()
