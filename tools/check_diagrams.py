#!/usr/bin/env python3
"""Validate the Mermaid blocks in the docs.

Two failure modes this catches, both of which render as a *plausible* diagram
rather than an error, so the eye does not catch them:

  COLLISION — the same node id declared twice with different labels. Mermaid
              silently keeps one and the other box vanishes from the picture
              while its edges remain, so the diagram quietly lies.

  DEPTH     — subgraph nesting beyond a readable limit, or an unbalanced
              subgraph/end pair, which shifts every following node into the
              wrong container.

Usage: tools/check_diagrams.py [file ...]   (default: docs/architecture.md)
"""
import re
import sys
import pathlib

MAX_SUBGRAPH_DEPTH = 2

# `id["label"]`, `id("label")`, `id{"label"}` — the declaration forms used here.
NODE_DECL = re.compile(r'(?:^|\s)([A-Za-z_][\w]*)\s*([\[\({])\s*"([^"]*)"')


def blocks(text):
    """Yield (start_line, [lines]) for each ```mermaid fence."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("```mermaid"):
            start = i + 1
            j = start
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            yield start, lines[start:j]
            i = j
        i += 1


def check(path):
    problems = []
    text = path.read_text()
    found = 0

    for start, body in blocks(text):
        found += 1
        labels = {}
        depth = 0
        max_depth = 0

        for offset, raw in enumerate(body):
            lineno = start + offset + 1
            line = raw.strip()
            if line.startswith("%%"):
                continue

            if re.match(r'^subgraph\b', line):
                depth += 1
                max_depth = max(max_depth, depth)
            elif line == "end":
                depth -= 1
                if depth < 0:
                    problems.append(
                        f"{path}:{lineno}: unbalanced 'end' — more ends than subgraphs")
                    depth = 0

            for node_id, _open, label in NODE_DECL.findall(raw):
                if node_id in labels and labels[node_id][0] != label:
                    problems.append(
                        f"{path}:{lineno}: COLLISION — node id '{node_id}' "
                        f"redeclared with a different label\n"
                        f"    first  (line {labels[node_id][1]}): {labels[node_id][0]!r}\n"
                        f"    second (line {lineno}): {label!r}")
                else:
                    labels.setdefault(node_id, (label, lineno))

        if depth != 0:
            problems.append(
                f"{path}: diagram starting line {start + 1} has {depth} "
                f"unclosed subgraph(s)")
        if max_depth > MAX_SUBGRAPH_DEPTH:
            problems.append(
                f"{path}: diagram starting line {start + 1} nests subgraphs "
                f"{max_depth} deep (max {MAX_SUBGRAPH_DEPTH})")

    return found, problems


def main():
    targets = sys.argv[1:] or ["docs/architecture.md"]
    total, all_problems = 0, []
    for t in targets:
        p = pathlib.Path(t)
        if not p.is_file():
            all_problems.append(f"{t}: not a file")
            continue
        found, problems = check(p)
        total += found
        all_problems += problems

    if all_problems:
        print("\n".join(all_problems), file=sys.stderr)
        print(f"\ndiagrams: {len(all_problems)} problem(s) in {total} diagram(s)",
              file=sys.stderr)
        return 1

    print(f"diagrams: OK ({total} checked, no id collisions, "
          f"nesting within {MAX_SUBGRAPH_DEPTH})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
