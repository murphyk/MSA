#!/usr/bin/env python3
"""Plot MSA posterior means (averaged over N runs) vs gold posterior means.

Reads MSA samples from `samples/{method}_r0/, _r1/, ...` and gold samples
from `samples/gold_e{expt}/`. Produces two figure files in
`comparisons/{method}_vs_gold_e{expt}/`:

  per_scenario.png   : 2x4 grid, one subplot per test scenario; each
                       subplot's 8 points are the 8 queries (MSA mean vs
                       gold mean).
  by_query_type.png  : 3 panels (constant q1-3, temporal q4-6,
                       new-match q7-8); each panel pools all T*Q points
                       for that query type, color-coded by scenario.

Usage:
    python plot_vs_gold.py --sport tug-of-war --method msa_flash_e1 --expt 1
    python plot_vs_gold.py --scenario tug-of-war_effort_team-... \
                           --method msa_flash_e1 --expt 1
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent

# Query partition (matches the 8-query structure of the test scenarios:
# Q1-3 are intrinsic-strength rank, Q4-6 are effort/accuracy in a specific
# match, Q7-8 are new-match predictions).
QUERY_GROUPS = [
    ('constant',    ['query1', 'query2', 'query3']),
    ('temporal',    ['query4', 'query5', 'query6']),
    ('prediction',  ['query7', 'query8']),
]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _suffix(expt):
    if not expt:
        return ''
    return f'_e{expt}' if str(expt).isdigit() else f'_{expt}'


def _is_probability(values):
    arr = np.asarray(values)
    return float(arr.max()) <= 1.0 and float(arr.min()) >= 0.0


def gather(scenario_id, method, expt):
    """Return per-query (msa_mean, msa_std_across_runs, gold_mean) for one
    test scenario. Probabilities are scaled to 0-100."""
    # MSA: load all runs matching {method}_r*
    method_pattern = method  # e.g. msa_flash_e1
    run_dirs = sorted((REPO / 'samples').glob(f'{method_pattern}_r*'))
    run_files = [d / f'{scenario_id}.json' for d in run_dirs
                 if (d / f'{scenario_id}.json').exists()]
    if not run_files:
        raise SystemExit(
            f"No MSA samples for {scenario_id} under {method_pattern}_r*")
    runs = [_load(p) for p in run_files]

    # Gold
    gold_path = REPO / 'samples' / f'gold{_suffix(expt)}' / f'{scenario_id}.json'
    if not gold_path.exists():
        raise SystemExit(f"Gold samples missing: {gold_path}")
    gold = _load(gold_path)

    queries = runs[0]['queries']
    out = []
    for q_key in queries:
        try:
            run_vals = [np.array([s[q_key] for s in r['samples']], float)
                        for r in runs]
            gold_vals = np.array([s[q_key] for s in gold['samples']], float)
        except KeyError:
            continue  # skip queries missing in any run
        pooled = np.concatenate(run_vals + [gold_vals])
        scale = 100.0 if _is_probability(pooled) else 1.0
        run_vals = [v * scale for v in run_vals]
        gold_vals = gold_vals * scale
        per_run_means = np.array([v.mean() for v in run_vals])
        out.append({
            'q_key': q_key,
            'label': queries[q_key],
            'msa_mean': float(per_run_means.mean()),
            'msa_std':  float(per_run_means.std(ddof=1)
                              if len(per_run_means) > 1 else 0.0),
            'gold_mean': float(gold_vals.mean()),
            'gold_sem':  float(gold_vals.std(ddof=1) /
                               np.sqrt(len(gold_vals))),
            'scale': scale,
        })
    return out


def _annotate_r2(ax, xs, ys):
    if len(xs) >= 2 and np.std(xs) > 0 and np.std(ys) > 0:
        r = float(np.corrcoef(xs, ys)[0, 1])
        ax.set_title(f"{ax.get_title()}\nR² = {r * r:.2f}", fontsize=9)


def plot_per_scenario(scenarios, all_rows, out_path):
    """2x4 grid: one subplot per scenario, points = the 8 queries."""
    n = len(scenarios)
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(13, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, scenario_id, rows_data in zip(axes, scenarios, all_rows):
        xs = [r['msa_mean'] for r in rows_data]
        ys = [r['gold_mean'] for r in rows_data]
        ax.errorbar(xs, ys,
                    xerr=[r['msa_std'] for r in rows_data],
                    yerr=[r['gold_sem'] for r in rows_data],
                    fmt='o', color='C4', ecolor='C4', capsize=2,
                    markersize=5)
        # Short title (drop sport prefix and trailing tags)
        short = scenario_id
        for prefix in ('biathalon_', 'canoe-race_', 'tug-of-war_'):
            short = short.removeprefix(prefix)
        short = short.replace('_dec_win_how_much_explicit_continuous_variable', '')
        short = short.replace('_dec_win_how_much_implicit_continuous_variable', '')
        # Truncate to keep panel readable
        if len(short) > 35:
            short = short[:32] + '...'
        ax.set_title(short, fontsize=8)
        _annotate_r2(ax, xs, ys)
    for ax in axes[n:]:
        ax.axis('off')
    # y = x reference + axis labels
    lo, hi = -5, 105
    for ax in axes[:n]:
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.4, lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)
    # Outer labels
    for ax in axes[(rows - 1) * cols:rows * cols]:
        ax.set_xlabel('MSA mean')
    for ax in axes[::cols]:
        ax.set_ylabel('Gold mean')
    fig.suptitle('Per-query posterior means: MSA (N runs) vs gold, '
                 'one panel per test scenario', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_by_query_type(scenarios, all_rows, out_path):
    """3 panels: constant / temporal / prediction. Each pools T scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))

    for ax, (group_name, q_keys) in zip(axes, QUERY_GROUPS):
        all_x, all_y = [], []
        for color, scenario_id, rows_data in zip(colors, scenarios, all_rows):
            xs = [r['msa_mean'] for r in rows_data if r['q_key'] in q_keys]
            ys = [r['gold_mean'] for r in rows_data if r['q_key'] in q_keys]
            xerr = [r['msa_std'] for r in rows_data if r['q_key'] in q_keys]
            yerr = [r['gold_sem'] for r in rows_data if r['q_key'] in q_keys]
            if xs:
                ax.errorbar(xs, ys, xerr=xerr, yerr=yerr,
                            fmt='o', color=color, ecolor=color,
                            capsize=2, markersize=5, alpha=0.85)
                all_x.extend(xs)
                all_y.extend(ys)
        ax.set_title(f'{group_name} ({len(all_x)} points)', fontsize=11)
        _annotate_r2(ax, np.array(all_x), np.array(all_y))
        ax.set_xlabel('MSA mean')

    axes[0].set_ylabel('Gold mean')
    lo, hi = -5, 105
    for ax in axes:
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.4, lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)

    fig.suptitle('Per-query posterior means by query type, pooled across '
                 f'T={len(scenarios)} test scenarios', y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--sport', help='produce both plots for all T=7 scenarios')
    g.add_argument('--scenario', help='single scenario (just 2x4 with one panel)')
    ap.add_argument('--method', required=True,
                    help='base MSA method, e.g. msa_flash_e1 '
                         '(N runs at {method}_r*)')
    ap.add_argument('--expt', default='1')
    args = ap.parse_args()

    if args.sport:
        test_ids = json.loads(
            (REPO / 'data' / 'test_ids.json').read_text())
        scenarios = [f'{args.sport}_{sfx}' for sfx in test_ids[args.sport]]
    else:
        scenarios = [args.scenario]

    all_rows = []
    for s in scenarios:
        all_rows.append(gather(s, args.method, args.expt))
        print(f"  loaded {s}")

    out_dir = REPO / 'comparisons' / f'{args.method}_vs_gold{_suffix(args.expt)}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sport:
        per_scenario_path = out_dir / f'{args.sport}_per_scenario.png'
        by_type_path = out_dir / f'{args.sport}_by_query_type.png'
    else:
        per_scenario_path = out_dir / f'{args.scenario}_per_scenario.png'
        by_type_path = out_dir / f'{args.scenario}_by_query_type.png'

    plot_per_scenario(scenarios, all_rows, per_scenario_path)
    print(f"Wrote {per_scenario_path.relative_to(REPO)}")
    plot_by_query_type(scenarios, all_rows, by_type_path)
    print(f"Wrote {by_type_path.relative_to(REPO)}")


if __name__ == '__main__':
    main()
