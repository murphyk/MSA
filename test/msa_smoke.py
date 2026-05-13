#!/usr/bin/env python3
"""Fast end-to-end smoke test for the MSA pipeline.

Runs msa.py on the first tug-of-war E1 test scenario with --llm flash,
--n-runs 1, --k-graph 1. ~3 LLM calls total + 1 webppl inference
(MSA) + 1 webppl inference (gold), finishes in well under a minute.

Outputs land at:
    models/msa_flash_e1_r0/{scenario}.wppl
    samples/msa_flash_e1_r0/{scenario}.json
    models/gold_e1/{scenario}.wppl
    samples/gold_e1/{scenario}.json
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    test_ids = json.loads((REPO / 'data' / 'test_ids.json').read_text())
    scenario = f'tug-of-war_{test_ids["tug-of-war"][0]}'
    print(f"Smoke test scenario: {scenario}")
    subprocess.run([
        sys.executable, str(REPO / 'msa.py'),
        '--scenario', scenario,
        '--llm', 'flash',
        '--n-runs', '1',
        '--k-graph', '1',
        '--expt', '1',
    ], check=True)


if __name__ == '__main__':
    main()
