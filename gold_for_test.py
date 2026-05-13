#!/usr/bin/env python3
"""Build a 'gold-for-this-test-scenario' WebPPL model.

The gold sport-knowledge model (models/gold/{sport}.wppl) hard-codes the
specific conditions and queries from its train scenario. To run gold
inference on a *test* scenario with different athletes and conditions,
we:

  1. Strip everything from `// CONDITIONS` to end of the model function
     out of the gold model.
  2. Append the LLM-parsed conditions (verbatim) and a return block
     synthesised from the LLM-parsed queries.
  3. Replace the trailing `var queryLabels = {...}` with one built from
     the test scenario's `Query N: ...` text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
GOLD_DIR = REPO / 'models' / 'gold'


def _extract_query_exprs(parse_block):
    """From a <START_LANGUAGE_TO_WEBPPL_CODE> block, return
    {query_number(int): webppl_expression(str)}.

    Recognises both formats produced by the LLM:
        Query 1: <NL question>            // or:  // Query 1: <NL>
          <webppl call>                   //       <webppl call>
    The expression line is the first non-comment, non-marker line after
    the query header.
    """
    in_queries = False
    out = {}
    current_q = None
    for raw in parse_block.splitlines():
        stripped = raw.strip()
        if stripped == '// QUERIES':
            in_queries = True
            continue
        if not in_queries:
            continue
        m = re.match(r'(?://\s*)?Query\s+(\d+)\s*:', stripped)
        if m:
            current_q = int(m.group(1))
            continue
        if (current_q is not None and stripped
                and not stripped.startswith('//')
                and not stripped.startswith('<')):
            out[current_q] = stripped.rstrip(',')
            current_q = None
    return out


def _extract_conditions_section(parse_block):
    """Pull just the // CONDITIONS section (without the //CONDITIONS header)
    from a <START_LANGUAGE_TO_WEBPPL_CODE> block."""
    lines = parse_block.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('// CONDITIONS') and start is None:
            start = i
        elif s.startswith('// QUERIES') and start is not None:
            end = i
            break
    if start is None:
        raise ValueError("No `// CONDITIONS` header in parse block")
    if end is None:
        # closing marker present?
        for i in range(start + 1, len(lines)):
            if lines[i].strip().startswith('<'):
                end = i
                break
        else:
            end = len(lines)
    return '\n'.join(lines[start:end]).rstrip()


def _build_query_labels(test_scenario_query):
    """Parse `Query N: <text>` lines into {queryN: text}."""
    labels = {}
    for ln in test_scenario_query.splitlines():
        m = re.match(r'\s*Query\s+(\d+)\s*:\s*(.+)', ln)
        if m:
            labels[f'query{m.group(1)}'] = m.group(2).strip()
    return labels


def build_gold_model(sport, parse_block, test_scenario_query):
    """Return WebPPL source for a gold model specialised to the test
    scenario's conditions/queries.

    sport: 'biathalon' | 'canoe-race' | 'tug-of-war' (matches gold filename)
    parse_block: output of step-1 (with start/end markers)
    test_scenario_query: the raw 'Query N: ...' text from the test scenario
    """
    gold_path = GOLD_DIR / f'{sport}.wppl'
    if not gold_path.exists():
        raise SystemExit(f"Gold model not found: {gold_path}")
    gold_src = gold_path.read_text()

    # Drop everything from "// CONDITIONS" onward in the gold file (which is
    # the hard-coded conditions, return block, closing }, and trailing
    # queryLabels). Keep the helper-function part of the model.
    idx = gold_src.find('// CONDITIONS')
    if idx < 0:
        raise SystemExit(f"Gold model missing '// CONDITIONS' marker: {gold_path}")
    knowledge = gold_src[:idx].rstrip()

    # New conditions section, from the LLM parse
    cond_section = _extract_conditions_section(parse_block)

    # New return block synthesised from the parsed query expressions
    exprs = _extract_query_exprs(parse_block)
    if not exprs:
        raise SystemExit("Failed to extract query expressions from parse block")
    items = ',\n        '.join(f'query{n}: {expr}'
                                for n, expr in sorted(exprs.items()))
    return_block = ('    return {\n        ' + items + '\n    }')

    # New queryLabels from the test scenario's question text
    labels = _build_query_labels(test_scenario_query)
    label_items = ',\n  '.join(f'{k}: {json.dumps(v)}' for k, v in labels.items())
    labels_block = (f'var queryLabels = {{\n  {label_items}\n}};\n')

    body = (
        f'{labels_block}\n'
        f'{knowledge}\n\n'
        f'    {cond_section}\n\n'
        f'{return_block}\n'
        '}\n'
    )
    return body
