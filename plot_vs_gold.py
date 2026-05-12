#!/usr/bin/env python3
"""Scatter per-query posterior means: MSA (across N runs) vs gold.

Reads samples from N independent runs of an MSA method
(samples/{method}_r0/, _r1/, ...) and the gold reference
(samples/{gold}/). For each query in the scenario:

    x = mean of pooled samples across all N runs
    y = mean of gold's samples
    xerr = std across the N per-run means (between-run variation)
    yerr = standard error of gold's posterior mean

If the query's samples all sit in [0, 1] we interpret them as probabilities
and multiply by 100 so the axes match the rank/percentage queries
(replicating Fig. 6 of Wong et al. 2025).

Usage:
    python plot_vs_gold.py --method msa_flash_e1 --scenario tug
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent


def load(path):
    with open(path) as f:
        return json.load(f)


def discover_runs(method, scenario):
    """Return list of sample-file paths for `{method}_r*/{scenario}.json`,
    falling back to a single `{method}/{scenario}.json`."""
    suffixed = sorted((REPO / 'samples').glob(f'{method}_r*'))
    paths = [d / f'{scenario}.json' for d in suffixed
             if (d / f'{scenario}.json').exists()]
    if paths:
        return paths
    single = REPO / 'samples' / method / f'{scenario}.json'
    if single.exists():
        return [single]
    raise SystemExit(f"No sample files found for method={method!r}, "
                     f"scenario={scenario!r}")


def is_probability_query(values):
    """Heuristic: if every observed sample is in [0, 1], scale to %."""
    arr = np.asarray(values)
    return float(arr.max()) <= 1.0 and float(arr.min()) >= 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True,
                    help='base method, e.g. msa_flash_e1 '
                         '(looks for {method}_r0, _r1, ...)')
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--gold-method', default='gold')
    args = ap.parse_args()

    run_paths = discover_runs(args.method, args.scenario)
    print(f"Found {len(run_paths)} run(s):")
    for p in run_paths:
        print(f"  {p.relative_to(REPO)}")

    runs = [load(p) for p in run_paths]
    gold = load(REPO / 'samples' / args.gold_method / f'{args.scenario}.json')

    queries = runs[0]['queries']

    rows = []
    for q_key in queries:
        run_vals = [np.array([s[q_key] for s in r['samples']], dtype=float)
                    for r in runs]
        gold_vals = np.array([s[q_key] for s in gold['samples']], dtype=float)

        # Auto-scale probabilities to percent so all queries share a 0-100 axis
        scale = 100.0 if is_probability_query(
            np.concatenate(run_vals + [gold_vals])) else 1.0
        run_vals = [v * scale for v in run_vals]
        gold_vals = gold_vals * scale

        per_run_means = np.array([v.mean() for v in run_vals])
        rows.append({
            'q_key': q_key,
            'x_mean': float(per_run_means.mean()),
            'x_std':  float(per_run_means.std(ddof=1)
                            if len(per_run_means) > 1 else 0.0),
            'y_mean': float(gold_vals.mean()),
            'y_sem':  float(gold_vals.std(ddof=1) / np.sqrt(len(gold_vals))),
            'scale':  scale,
        })

    x = np.array([r['x_mean'] for r in rows])
    y = np.array([r['y_mean'] for r in rows])
    xerr = np.array([r['x_std'] for r in rows])
    yerr = np.array([r['y_sem'] for r in rows])

    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else float('nan')
    r2 = r ** 2 if np.isfinite(r) else float('nan')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color='C4',
                ecolor='C4', capsize=3, markersize=7)
    for row, xi, yi in zip(rows, x, y):
        ax.annotate(row['q_key'], (xi, yi), xytext=(6, 6),
                    textcoords='offset points', fontsize=9, color='C4')

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = max(5.0, (hi - lo) * 0.15)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5, label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(f"MSA mean ({args.method}, N={len(run_paths)} runs)")
    ax.set_ylabel(f"Gold mean ({args.gold_method})")
    ax.set_title(f"{args.scenario}: per-query posterior means — "
                 f"MSA vs Gold  (R² = {r2:.3f})")
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_dir = REPO / 'comparisons' / f'{args.method}_vs_{args.gold_method}_scatter'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.scenario}.png'
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(REPO)}  (R² = {r2:.3f})")

    # Also dump the per-query numbers next to the plot
    csv_path = out_dir / f'{args.scenario}.csv'
    with open(csv_path, 'w') as f:
        f.write('query,x_mean,x_std,y_mean,y_sem,scale\n')
        for row in rows:
            f.write(f"{row['q_key']},{row['x_mean']:.4f},{row['x_std']:.4f},"
                    f"{row['y_mean']:.4f},{row['y_sem']:.4f},{row['scale']}\n")
    print(f"Wrote {csv_path.relative_to(REPO)}")


if __name__ == '__main__':
    main()
