#!/usr/bin/env python3
"""Generate docs/human_data.html: a browseable summary of
data/msa_cogsci_human_data.json (Wong et al. 2025 human responses)."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

REPO = Path(__file__).resolve().parent
DATA = REPO / 'data' / 'msa_cogsci_human_data.json'
OUT = REPO / 'docs' / 'human_data.html'


def per_participant_mean(participant_dict, qkey):
    """Mean across a participant's clicks for one query."""
    vals = participant_dict.get(qkey, [])
    return mean(vals) if vals else None


def summarize_query(vignette_data, qkey):
    """Across participants in a vignette, return (mean, std, n) of their
    per-participant means for query qkey."""
    pmeans = [per_participant_mean(p, qkey) for p in vignette_data]
    pmeans = [m for m in pmeans if m is not None]
    if not pmeans:
        return None, None, 0
    m = mean(pmeans)
    s = pstdev(pmeans) if len(pmeans) > 1 else 0.0
    return m, s, len(pmeans)


def render_table(experiment_data):
    """Return an HTML <table> for one experiment."""
    rows = []
    for key, vignette in sorted(experiment_data.items()):
        sport = key.split('_')[0]
        n_part = len(vignette)
        qkeys = sorted({q for p in vignette for q in p}, key=lambda k: int(k[5:]))
        cells = []
        for qkey in qkeys:
            m, s, _ = summarize_query(vignette, qkey)
            cells.append(f"{m:.1f}<span class='s'> &plusmn;{s:.1f}</span>"
                         if m is not None else '&mdash;')
        rows.append((sport, key, n_part, qkeys, cells))

    if not rows:
        return ''

    nq = max(len(r[3]) for r in rows)
    parts = ['<table>', '<thead><tr><th>sport</th><th>vignette</th><th>N</th>']
    for i in range(nq):
        parts.append(f'<th>q{i+1}</th>')
    parts.append('</tr></thead>\n<tbody>')

    for sport, key, n_part, qkeys, cells in rows:
        parts.append(f"<tr><td class='sport-{sport}'>{sport}</td>"
                     f"<td class='key'><span>{key}</span></td>"
                     f"<td class='n'>{n_part}</td>")
        for c in cells:
            parts.append(f"<td>{c}</td>")
        for _ in range(nq - len(cells)):
            parts.append('<td>&mdash;</td>')
        parts.append('</tr>')
    parts.append('</tbody></table>')
    return '\n'.join(parts)


def main():
    data = json.loads(DATA.read_text())

    # Summary stats
    summary_lines = []
    for exp, vignettes in data.items():
        sports = Counter(k.split('_')[0] for k in vignettes)
        pcounts = [len(v) for v in vignettes.values()]
        qcounts = [len(v[0]) for v in vignettes.values() if v]
        click_counts = [len(p[next(iter(p))]) for v in vignettes.values()
                        for p in v]
        summary_lines.append(
            f"<li><b>{exp}</b>: {len(vignettes)} vignettes "
            f"({', '.join(f'{n} {s}' for s, n in sports.items())}), "
            f"{min(pcounts)}&ndash;{max(pcounts)} participants/vignette, "
            f"{qcounts[0]} queries/vignette, "
            f"{min(click_counts)}&ndash;{max(click_counts)} clicks/query.</li>")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Wong et al. 2025 &mdash; human data</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       max-width: 1300px; margin: 1.5em auto; padding: 0 1em; color: #222;
       line-height: 1.5; }}
h1 {{ margin-bottom: 0.2em; }}
h2 {{ margin-top: 2em; }}
table {{ border-collapse: collapse; margin: 0.5em 0 2em; font-size: 0.92em; }}
th, td {{ border: 1px solid #ccc; padding: 0.3em 0.5em; text-align: right; }}
th {{ background: #f0f0f0; }}
td.key {{ text-align: left; padding: 0; width: 460px; max-width: 460px; }}
td.key span {{ display: block; padding: 0.3em 0.5em;
               font-family: "SF Mono", Menlo, monospace; font-size: 0.85em;
               white-space: nowrap; overflow-x: auto; }}
td.n {{ color: #666; }}
.s {{ color: #888; font-size: 0.85em; }}
td.sport-canoe-race  {{ background: #eef8ee; }}
td.sport-biathalon   {{ background: #eef0fa; }}
td.sport-tug-of-war  {{ background: #faeeee; }}
.note {{ background: #fff8e1; border-left: 4px solid #f1c40f;
         padding: 0.6em 1em; margin: 1em 0; }}
ul {{ line-height: 1.7; }}
</style></head><body>

<h1>Wong et al. 2025 &mdash; human response data</h1>
<p>Source: <code>data/msa_cogsci_human_data.json</code></p>

<h2>Summary</h2>
<ul>
{chr(10).join(summary_lines)}
</ul>
<p>Each cell shows the across-participant mean &plusmn;std of each participant's
mean click for that query, on a 0&ndash;100 response scale. <i>N</i> is the
number of participants for that vignette. Hover the vignette name to see
the full key.</p>

<div class='note'>
The vignette key encodes scenario structure
(e.g. <code>team-confounded-opponent-12</code>,
<code>team-explain-away-11</code>). Our codebase currently has just one
canonical vignette per sport (in <code>scenarios/</code>); the human data
covers seven structural variants per sport per experiment.
</div>

<h2>e1_explicit &mdash; full background</h2>
{render_table(data['e1_explicit'])}

<h2>e2_implicit &mdash; under-specified background</h2>
{render_table(data['e2_implicit'])}

</body></html>
"""
    OUT.write_text(html)
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == '__main__':
    main()
