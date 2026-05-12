#!/usr/bin/env python3
"""Compare posterior samples produced by two methods, for one model/experiment.

Reads:
    samples/{method1}/{model}[_e{expt}].json
    samples/{method2}/{model}[_e{expt}].json

Writes (one PNG per query, one JSD summary, one HTML index):
    comparisons/{method1}_vs_{method2}/{model}[_e{expt}]/query{q}.png
    comparisons/{method1}_vs_{method2}/{model}[_e{expt}]/jsd.png
    comparisons/{method1}_vs_{method2}/{model}[_e{expt}]/comparison.html

Usage:
    python compare_samples.py --model tug --method1 gold --method2 gold1
    python compare_samples.py --model tug --method1 gold --method2 gold1 --expt 3
"""

import argparse
import json
import os

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, permutation_test, wasserstein_distance


def jensen_shannon_distance(vals1, vals2, n_bins=30):
    """Bin-based Jensen-Shannon distance (sqrt of JS divergence) in [0, 1]."""
    combined = np.concatenate([vals1, vals2])
    edges = np.histogram_bin_edges(combined, bins=n_bins)
    p, _ = np.histogram(vals1, bins=edges, density=False)
    q, _ = np.histogram(vals2, bins=edges, density=False)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.sqrt(max(jsd, 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--method1', required=True)
    ap.add_argument('--method2', required=True)
    ap.add_argument('--expt', default='')
    ap.add_argument('--bins', type=int, default=30)
    ap.add_argument('--n-perm', type=int, default=1000,
                    help='permutation resamples for Wasserstein p-value')
    args = ap.parse_args()

    expt_suffix = f"_e{args.expt}" if args.expt else ""
    expt_str = args.expt if args.expt else '[]'

    file1 = os.path.join('samples', args.method1, f"{args.model}{expt_suffix}.json")
    file2 = os.path.join('samples', args.method2, f"{args.model}{expt_suffix}.json")
    outdir = os.path.join('comparisons', f"{args.method1}_vs_{args.method2}",
                          f"{args.model}{expt_suffix}")

    with open(file1) as f:
        d1 = json.load(f)
    with open(file2) as f:
        d2 = json.load(f)

    if d1['queries'] != d2['queries']:
        raise SystemExit(f"Query sets differ between {file1} and {file2}")

    queries = d1['queries']
    samples1 = d1['samples']
    samples2 = d2['samples']

    os.makedirs(outdir, exist_ok=True)
    header = (f"Model={args.model}, method1={args.method1}, "
              f"method2={args.method2}, expt={expt_str}")

    png_names = []
    rows = []
    rng = np.random.default_rng(0)
    for q_key, q_text in queries.items():
        vals1 = np.array([s[q_key] for s in samples1], dtype=float)
        vals2 = np.array([s[q_key] for s in samples2], dtype=float)

        jsd = jensen_shannon_distance(vals1, vals2, n_bins=args.bins)
        w = wasserstein_distance(vals1, vals2)
        pooled = np.concatenate([vals1, vals2])
        rng_span = float(pooled.max() - pooled.min())
        pooled_sd = float(pooled.std(ddof=1))
        w_norm = float(w / rng_span) if rng_span > 0 else 0.0
        w_std = float(w / pooled_sd) if pooled_sd > 0 else 0.0
        ks_stat, ks_p = ks_2samp(vals1, vals2)
        perm = permutation_test(
            (vals1, vals2),
            statistic=lambda a, b: wasserstein_distance(a, b),
            alternative='greater',
            n_resamples=args.n_perm,
            random_state=rng,
        )
        perm_p = float(perm.pvalue)
        rows.append({
            'query': q_key, 'label': q_text,
            'wasserstein': w, 'w_norm': w_norm, 'w_std': w_std,
            'jsd': jsd,
            'ks_stat': float(ks_stat), 'ks_p': float(ks_p),
            'perm_p': perm_p,
        })

        combined = np.concatenate([vals1, vals2])
        edges = np.histogram_bin_edges(combined, bins=args.bins)

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)
        axes[0].hist(vals1, bins=edges, color='C0', alpha=0.85)
        axes[0].set_ylabel('count')
        axes[0].set_title(args.method1, fontsize=10)
        axes[1].hist(vals2, bins=edges, color='C1', alpha=0.85)
        axes[1].set_ylabel('count')
        axes[1].set_xlabel(q_key)
        axes[1].set_title(args.method2, fontsize=10)
        fig.suptitle(
            f"{q_text}\n{header}\n"
            f"W={w:.3f}  JS={jsd:.3f}  KS p={ks_p:.3f}  perm p={perm_p:.3f}"
        )

        outpath = os.path.join(outdir, f"{q_key}.png")
        fig.tight_layout()
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        png_names.append(os.path.basename(outpath))
        print(f"Wrote {outpath}  (W={w:.3f}, JS={jsd:.3f}, "
              f"KS p={ks_p:.3f}, perm p={perm_p:.3f})")

    jsd_scores = {r['query']: r['jsd'] for r in rows}

    csv_path = os.path.join(outdir, "metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    pval_path = os.path.join(outdir, "pvalues.png")
    edges = np.linspace(0, 1, 11)
    ks_pvals = [r['ks_p'] for r in rows]
    perm_pvals = [r['perm_p'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, vals, name in [(axes[0], ks_pvals, 'KS'),
                           (axes[1], perm_pvals, 'permutation on W')]:
        ax.hist(vals, bins=edges, color='C3', alpha=0.85, edgecolor='white')
        ax.axhline(len(rows) / 10, color='gray', ls='--', lw=1,
                   label='uniform expectation')
        ax.set_xlim(0, 1)
        ax.set_xlabel('p-value')
        ax.set_title(f"{name}  (n queries = {len(rows)})")
        ax.legend(fontsize=8)
    axes[0].set_ylabel('count')
    fig.suptitle(f"P-value distribution across queries\n{header}")
    fig.tight_layout()
    fig.savefig(pval_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {pval_path}")

    jsd_path = os.path.join(outdir, "jsd.png")
    keys = list(jsd_scores.keys())
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(keys) + 2), 4))
    ax.bar(keys, [jsd_scores[k] for k in keys], color='C2')
    ax.set_ylabel('Jensen-Shannon distance')
    ax.set_xlabel('query')
    ax.set_ylim(0, max(0.3, max(jsd_scores.values()) * 1.15))
    ax.set_title(f"JS distance per query\n{header}")
    for k, v in jsd_scores.items():
        ax.text(k, v, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(jsd_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {jsd_path}")

    html_path = os.path.join(outdir, "comparison.html")
    with open(html_path, 'w') as f:
        f.write("<!doctype html>\n<html><head><meta charset='utf-8'>\n")
        f.write(f"<title>{header}</title>\n")
        f.write("<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;padding:0 1em}"
                "img{max-width:100%;display:block;margin:1em 0;border:1px solid #ddd}"
                "h1{margin-bottom:0}p.sub{color:#666;margin-top:0.2em}"
                "table{border-collapse:collapse;margin:1em 0;font-size:0.95em}"
                "th,td{border:1px solid #ccc;padding:0.3em 0.7em;text-align:right}"
                "th{background:#f0f0f0}td:nth-child(2){text-align:left}</style>\n")
        f.write("</head><body>\n")
        f.write(f"<h1>{args.model} &mdash; {args.method1} vs {args.method2}</h1>\n")
        f.write(f"<p class='sub'>expt = {expt_str}</p>\n")
        f.write(f"<img src='{os.path.basename(pval_path)}' alt='p-value distribution'>\n")
        f.write(f"<img src='{os.path.basename(jsd_path)}' alt='JS distance summary'>\n")
        f.write("<h2>Per-query metrics</h2>\n<table>\n")
        f.write("<tr><th>query</th><th>label</th><th>W</th><th>W/range</th><th>W/sd</th>"
                "<th>JS</th><th>KS stat</th><th>KS p</th><th>perm p</th></tr>\n")
        for r in rows:
            f.write(f"<tr><td>{r['query']}</td><td>{r['label']}</td>"
                    f"<td>{r['wasserstein']:.3f}</td>"
                    f"<td>{r['w_norm']:.3f}</td><td>{r['w_std']:.3f}</td>"
                    f"<td>{r['jsd']:.3f}</td>"
                    f"<td>{r['ks_stat']:.3f}</td><td>{r['ks_p']:.3f}</td>"
                    f"<td>{r['perm_p']:.3f}</td></tr>\n")
        f.write("</table>\n")
        for name in png_names:
            f.write(f"<img src='{name}' alt='{name}'>\n")
        f.write("</body></html>\n")
    print(f"Wrote {html_path}")


if __name__ == '__main__':
    main()
