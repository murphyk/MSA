#!/usr/bin/env python3
"""Build data/test_ids.json from the human-data file.

Output schema:
    {
        "biathalon":   [<id_suffix>, ...],   # 7 entries
        "canoe-race":  [<id_suffix>, ...],
        "tug-of-war":  [<id_suffix>, ...]
    }

The id_suffix has the sport prefix stripped, so to find the corresponding
test scenario file you reconstruct
    data/test-scenarios/e1/scenarios/{sport}_{id_suffix}.txt
"""

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
HUMAN = REPO / 'data' / 'msa_cogsci_human_data.json'
OUT = REPO / 'data' / 'test_ids.json'


def main():
    data = json.loads(HUMAN.read_text())
    by_sport = defaultdict(list)
    # E1 and E2 share the same vignette set; use e1 keys.
    for full_id in data['e1_explicit']:
        sport, _, suffix = full_id.partition('_')
        by_sport[sport].append(suffix)
    for s in by_sport:
        by_sport[s].sort()

    OUT.write_text(json.dumps(dict(sorted(by_sport.items())), indent=2) + '\n')
    print(f"Wrote {OUT.relative_to(REPO)}")
    for sport, ids in sorted(by_sport.items()):
        print(f"  {sport}: {len(ids)} ids")


if __name__ == '__main__':
    main()
