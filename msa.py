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
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent


def formalize(scenario_name: str, expt: str, llm: str) -> str:
    """Return the WebPPL source code for the formalized scenario.

    MOCK: copies the corresponding gold model verbatim. The real
    implementation will call out to {llm} with the scenario text and
    return the LLM-produced WebPPL code, varying with `expt`.
    """
    gold_path = REPO / 'models' / 'gold' / f'{scenario_name}.wppl'
    if not gold_path.exists():
        raise SystemExit(f"Gold model not found: {gold_path}")
    return gold_path.read_text()


def run(cmd):
    print('$ ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True,
                    help='scenario name (e.g. tug)')
    ap.add_argument('--expt', default='1',
                    help='experiment label (default: 1)')
    ap.add_argument('--llm', default='pro',
                    help='llm label, e.g. pro for Gemini 3.1 Pro (default: pro)')
    args = ap.parse_args()

    expt_suffix = f'_e{args.expt}' if args.expt else ''
    method = f'msa_{args.llm}{expt_suffix}'

    # 1. Formalize and save model.
    model_src = formalize(args.scenario, args.expt, args.llm)
    model_dir = REPO / 'models' / method
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'{args.scenario}.wppl'
    model_path.write_text(model_src)
    print(f"Wrote {model_path.relative_to(REPO)}")

    # 2. Run inference. get_samples.js defaults --model-dir to models/{method}.
    run(['node', str(REPO / 'get_samples.js'),
         '--model', args.scenario,
         '--method', method])

    # 3. Compare against gold1.
    run([sys.executable, str(REPO / 'compare_samples.py'),
         '--model', args.scenario,
         '--method1', 'gold1',
         '--method2', method])


if __name__ == '__main__':
    main()
