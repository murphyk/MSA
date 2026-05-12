#!/usr/bin/env python3
"""Parse scenarios/{model}.txt into scenarios/{model}.json.

Each .txt file is annotated with marker lines. This script extracts the
content between each pair and emits a JSON object with the fields:

    scenario          <START_SCENARIO>            ... <END_SCENARIO>
    language_to_code  <START_LANGUAGE_TO_WEBPPL_CODE> ... <END_LANGUAGE_TO_WEBPPL_CODE>
    informal          <START_SCRATCHPAD>          ... <START_CONCEPT_TRACE>
    graph             <START_CONCEPT_TRACE>       ... <END_CONCEPT_TRACE>
    model             <START_WEBPPL_MODEL>        ... <END_WEBPPL_MODEL>
                      (trailing inference/viz lines stripped)
"""

import json
from pathlib import Path

SCENARIO_DIR = Path('scenarios')

SECTIONS = [
    ('<START_SCENARIO>', '<END_SCENARIO>', 'scenario'),
    ('<START_LANGUAGE_TO_WEBPPL_CODE>', '<END_LANGUAGE_TO_WEBPPL_CODE>', 'language_to_code'),
    ('<START_SCRATCHPAD>', '<START_CONCEPT_TRACE>', 'informal'),
    ('<START_CONCEPT_TRACE>', '<END_CONCEPT_TRACE>', 'graph'),
    ('<START_WEBPPL_MODEL>', '<END_WEBPPL_MODEL>', 'model'),
]


def extract(text, start, end):
    s = text.find(start)
    if s < 0:
        raise ValueError(f"Missing marker: {start}")
    s += len(start)
    e = text.find(end, s)
    if e < 0:
        raise ValueError(f"Missing marker: {end}")
    return text[s:e].strip('\n')


def strip_inference(model_src):
    """Drop the trailing `// Now we run the model ...` / `Infer` / `viz` block."""
    lines = model_src.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('// Now we run the model'):
            return '\n'.join(lines[:i]).rstrip()
    return model_src.rstrip()


def parse(text):
    """Return one dict per scenario.

    Each field is stored as a list of lines (so the JSON renders readably
    in editors). Reconstruct the original text with '\\n'.join(field).
    """
    out = {}
    for start, end, key in SECTIONS:
        block = extract(text, start, end)
        if key == 'model':
            block = strip_inference(block)
        out[key] = block.split('\n')
    return out


def main():
    txt_files = sorted(SCENARIO_DIR.glob('*.txt'))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {SCENARIO_DIR}/")
    for txt_path in txt_files:
        parsed = parse(txt_path.read_text())
        json_path = txt_path.with_suffix('.json')
        json_path.write_text(json.dumps(parsed, indent=2) + '\n')
        print(f"Wrote {json_path}  "
              f"(scenario={len(parsed['scenario'])} lines, "
              f"informal={len(parsed['informal'])} lines, "
              f"graph={len(parsed['graph'])} lines, "
              f"model={len(parsed['model'])} lines)")


if __name__ == '__main__':
    main()
