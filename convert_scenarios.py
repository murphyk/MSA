#!/usr/bin/env python3
"""Convert scenarios/{name}.txt into scenarios/{name}.json.

Each input file is annotated with marker lines around five regions; some
regions have a further heading structure that we split out.

Output fields (each stored as a list of lines so the JSON renders readably):

    background, conditions, query
        From <START_SCENARIO> ... <END_SCENARIO>, split on the
        BACKGROUND / CONDITIONS / QUERIES headings.
    parsed_conditions, parsed_queries
        From <START_LANGUAGE_TO_WEBPPL_CODE> ... <END_LANGUAGE_TO_WEBPPL_CODE>,
        split on the // CONDITIONS / // QUERIES headings.
    informal
        From <START_SCRATCHPAD> up to <START_CONCEPT_TRACE>.
    graph
        From <START_CONCEPT_TRACE> to <END_CONCEPT_TRACE>.
    model
        From <START_WEBPPL_MODEL> to <END_WEBPPL_MODEL>, with the trailing
        `// Now we run the model ...` / Infer / viz block stripped.
"""

import json
from pathlib import Path

SCENARIO_DIR = Path('scenarios')


# Hand-crafted under-specified backgrounds for Experiment 2 (Wong et al. 2025).
# Each names only the latent variables that appear in the queries/conditions
# (e.g. intrinsic strength, effort), and drops the description of how those
# variables combine to produce observed outcomes. The official e2 vignettes
# were not published in the data repo; these are an approximation. The author
# was emailed for the official text.
BACKGROUND_E2 = {
    'tug': [
        "In this event, athletes are competing in matches of tug-of-war. "
        "Each athlete has an intrinsic strength, and puts in some level of "
        "effort to each match.",
        "Athletes compete either individually or as a team.",
        "All matches take place on the same day.",
    ],
    'biathlon': [
        "In this event, teams of players are competing in rounds of a "
        "biathalon, a winter sport that combines cross-country skiing and "
        "rifle shooting. Each athlete has an intrinsic strength, and has "
        "some shooting accuracy in each round.",
        "Athletes compete either individually or as a team.",
        "All matches take place on the same day.",
    ],
    'canoe': [
        "In this event, athletes are competing in a series of canoe races. "
        "Each athlete has an intrinsic strength, and puts in some level of "
        "effort to each race.",
        "Athletes compete either individually or as a team.",
        "All races take place on the same day.",
    ],
    'diving': [
        "In this event, athletes are competing in a series of synchronized "
        "diving tournaments. Each athlete has an intrinsic skill, and there "
        "is some level of synchronization between teammates in each round.",
        "Athletes compete as part of a team.",
        "All matches take place on the same day.",
    ],
    'exam': [
        "In this model, students are being evaluated for their science class, "
        "which has a two-part exam. Exams are completed in pairs.",
        "Each student has an intrinsic memorization ability, and has some "
        "level of laboratory performance on each evaluation.",
    ],
}


def extract(text, start, end):
    s = text.find(start)
    if s < 0:
        raise ValueError(f"Missing marker: {start}")
    s += len(start)
    e = text.find(end, s)
    if e < 0:
        raise ValueError(f"Missing marker: {end}")
    return text[s:e].strip('\n')


def split_on_headings(text, headings):
    """Split `text` into named sections.

    headings: list of (heading_line, output_key) pairs, in the order they
    appear. Returns {key: list-of-lines} with each heading line itself
    removed and leading/trailing blank lines trimmed.
    """
    lines = text.splitlines()
    indices = []
    for heading, _ in headings:
        target = heading.strip()
        for i, line in enumerate(lines):
            if line.strip() == target:
                indices.append(i)
                break
        else:
            raise ValueError(f"Heading not found: {heading!r}")
    indices.append(len(lines))

    out = {}
    for (_, key), start, end in zip(headings, indices, indices[1:]):
        section = lines[start + 1:end]
        while section and not section[0].strip():
            section.pop(0)
        while section and not section[-1].strip():
            section.pop()
        out[key] = section
    return out


def strip_inference(model_src):
    """Drop the trailing `// Now we run the model ...` / Infer / viz block."""
    lines = model_src.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('// Now we run the model'):
            return '\n'.join(lines[:i]).rstrip()
    return model_src.rstrip()


def parse(text):
    out = {}

    scenario = extract(text, '<START_SCENARIO>', '<END_SCENARIO>')
    out.update(split_on_headings(scenario, [
        ('BACKGROUND', 'background'),
        ('CONDITIONS', 'conditions'),
        ('QUERIES',    'query'),
    ]))

    l2c = extract(text, '<START_LANGUAGE_TO_WEBPPL_CODE>',
                          '<END_LANGUAGE_TO_WEBPPL_CODE>')
    out.update(split_on_headings(l2c, [
        ('// CONDITIONS', 'parsed_conditions'),
        ('// QUERIES',    'parsed_queries'),
    ]))

    informal = extract(text, '<START_SCRATCHPAD>', '<START_CONCEPT_TRACE>')
    out['informal'] = informal.splitlines()

    graph = extract(text, '<START_CONCEPT_TRACE>', '<END_CONCEPT_TRACE>')
    out['graph'] = graph.splitlines()

    model = extract(text, '<START_WEBPPL_MODEL>', '<END_WEBPPL_MODEL>')
    out['model'] = strip_inference(model).splitlines()

    return out


def main():
    txt_files = sorted(SCENARIO_DIR.glob('*.txt'))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {SCENARIO_DIR}/")
    for txt_path in txt_files:
        parsed = parse(txt_path.read_text())
        name = txt_path.stem
        if name in BACKGROUND_E2:
            parsed['background_e2'] = BACKGROUND_E2[name]
        json_path = txt_path.with_suffix('.json')
        json_path.write_text(json.dumps(parsed, indent=2) + '\n')
        sizes = ', '.join(f"{k}={len(v)}" for k, v in parsed.items())
        print(f"Wrote {json_path}  ({sizes} lines)")


if __name__ == '__main__':
    main()
