#!/usr/bin/env python3
"""Compare two posterior-sample JSON files (as produced by get_samples.js)
by plotting a stacked histogram per query and reporting a similarity score.

Usage:
    python compare_samples.py --model tug \
        --file1 gold_samples/tug_n1000_seed0.json \
        --file2 gold_samples/tug_n1000_seed1.json
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


def jensen_shannon_distance(vals1, vals2, n_bins=30):
    """Bin-based Jensen-Shannon distance (sqrt of JS divergence) in [0, 1].

    0 = identical histograms, 1 = disjoint supports.
    """
    combined = np.concatenate([vals1, vals2])
    edges = np.histogram_bin_edges(combined, bins=n_bins)
    p, _ = np.histogram(vals1, bins=edges, density=False)
    q, _ = np.histogram(vals2, bins=edges, density=False)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)  # in bits, in [0, 1]
    return float(np.sqrt(max(jsd, 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--file1', required=True)
    ap.add_argument('--file2', required=True)
    ap.add_argument('--outdir', default='figs')
    ap.add_argument('--bins', type=int, default=30)
    args = ap.parse_args()

    with open(args.file1) as f:
        d1 = json.load(f)
    with open(args.file2) as f:
        d2 = json.load(f)

    if d1['queries'] != d2['queries']:
        raise SystemExit(f"Query sets differ between {args.file1} and {args.file2}")

    queries = d1['queries']
    samples1 = d1['samples']
    samples2 = d2['samples']

    os.makedirs(args.outdir, exist_ok=True)
    label1 = Path(args.file1).stem
    label2 = Path(args.file2).stem

    png_names = []
    jsd_scores = {}
    for q_key, q_text in queries.items():
        vals1 = np.array([s[q_key] for s in samples1], dtype=float)
        vals2 = np.array([s[q_key] for s in samples2], dtype=float)

        w = wasserstein_distance(vals1, vals2)
        jsd = jensen_shannon_distance(vals1, vals2, n_bins=args.bins)
        jsd_scores[q_key] = jsd

        combined = np.concatenate([vals1, vals2])
        edges = np.histogram_bin_edges(combined, bins=args.bins)

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)
        axes[0].hist(vals1, bins=edges, color='C0', alpha=0.85)
        axes[0].set_ylabel('count')
        axes[0].set_title(label1, fontsize=10)
        axes[1].hist(vals2, bins=edges, color='C1', alpha=0.85)
        axes[1].set_ylabel('count')
        axes[1].set_xlabel(q_key)
        axes[1].set_title(label2, fontsize=10)
        fig.suptitle(f"{q_text}\nJS dist = {jsd:.3f}  |  Wasserstein = {w:.3f}")

        outpath = os.path.join(args.outdir, f"{args.model}_{q_key}.png")
        fig.tight_layout()
        fig.savefig(outpath, dpi=120)
        plt.close(fig)
        png_names.append(os.path.basename(outpath))
        print(f"Wrote {outpath}  (JS={jsd:.3f}, Wasserstein={w:.3f})")

    jsd_path = os.path.join(args.outdir, f"{args.model}_jsdist.png")
    keys = list(jsd_scores.keys())
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(keys) + 2), 4))
    ax.bar(keys, [jsd_scores[k] for k in keys], color='C2')
    ax.set_ylabel('Jensen-Shannon distance')
    ax.set_xlabel('query')
    ax.set_ylim(0, max(0.3, max(jsd_scores.values()) * 1.15))
    ax.set_title(f"{args.model}: JS distance per query  ({label1} vs {label2})")
    for k, v in jsd_scores.items():
        ax.text(k, v, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(jsd_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {jsd_path}")

    html_path = os.path.join(args.outdir, f"{args.model}_comparison.html")
    with open(html_path, 'w') as f:
        f.write(f"<!doctype html>\n<html><head><meta charset='utf-8'>\n")
        f.write(f"<title>{args.model} comparison</title>\n")
        f.write("<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;padding:0 1em}"
                "img{max-width:100%;display:block;margin:1em 0;border:1px solid #ddd}"
                "h1{margin-bottom:0}p.sub{color:#666;margin-top:0.2em}</style>\n")
        f.write("</head><body>\n")
        f.write(f"<h1>{args.model} &mdash; sample comparison</h1>\n")
        f.write(f"<p class='sub'>{label1} vs {label2}</p>\n")
        f.write(f"<img src='{os.path.basename(jsd_path)}' alt='JS distance summary'>\n")
        for name in png_names:
            f.write(f"<img src='{name}' alt='{name}'>\n")
        f.write("</body></html>\n")
    print(f"Wrote {html_path}")


if __name__ == '__main__':
    main()
