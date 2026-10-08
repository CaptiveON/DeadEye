#!/usr/bin/env python3
"""Static checks for the paper when no TeX toolchain is available: braces, environments, cite keys, inputs."""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "paper"
bib_keys = set(re.findall(r"^@\w+\{([^,\s]+),", (root / "refs.bib").read_text(), flags=re.M))
problems = []
for tex in sorted((root / "sections").glob("*.tex")) + [root / "main.tex"]:
    s = tex.read_text()
    body = re.sub(r"(?<!\\)%.*", "", s)  # strip comments
    verb_free = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", body, flags=re.S)
    depth = 0
    for ch in verb_free.replace("\\{", "").replace("\\}", ""):
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            break
    if depth != 0:
        problems.append(f"{tex.name}: unbalanced braces (net {depth})")
    begins = re.findall(r"\\begin\{(\w+\*?)\}", body)
    ends = re.findall(r"\\end\{(\w+\*?)\}", body)
    for env in set(begins + ends):
        if begins.count(env) != ends.count(env):
            problems.append(f"{tex.name}: environment {env} opened {begins.count(env)}x, closed {ends.count(env)}x")
    for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}", verb_free):
        for key in m.group(1).split(","):
            key = key.strip()
            if key and key not in bib_keys:
                problems.append(f"{tex.name}: unknown cite key {key}")
    for m in re.finditer(r"\\input\{([^}]*)\}", body):
        target = root / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        if not target.exists() and not m.group(1).startswith("tables/"):
            problems.append(f"{tex.name}: missing input {m.group(1)}")
print(f"{len(bib_keys)} bib keys; {len(problems)} problems")
for p in problems:
    print(" -", p)
sys.exit(1 if problems else 0)
