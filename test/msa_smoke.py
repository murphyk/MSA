#!/usr/bin/env python3
"""Fast end-to-end smoke test for the MSA pipeline.

Runs msa.py on the tug scenario with --llm flash and --k-graph 1, so it
only needs ~3 LLM calls (parse + 1 graph + score + model) and finishes
in well under a minute. Use this for iterating on pipeline changes
without spending Pro tokens.

Outputs land at:
    models/msa_flash_esmoke/tug.wppl
    samples/msa_flash_esmoke/tug.json
    comparisons/gold1_vs_msa_flash_esmoke/tug/

Usage:
    python test/msa_smoke.py
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    subprocess.run([
        sys.executable, str(REPO / 'msa.py'),
        '--scenario', 'tug',
        '--llm', 'flash',
        '--k-graph', '1',
        '--expt', 'smoke',
    ], check=True)


if __name__ == '__main__':
    main()
