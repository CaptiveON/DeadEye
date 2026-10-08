"""Robust extraction of an action from free-form model output."""
from __future__ import annotations

import difflib
import re

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
CLOSE_THINK_RE = re.compile(r"\A(.*)</think>", re.DOTALL | re.IGNORECASE)
# `Action: B` and also JSON-style `"action": "B"` (the quote closing the key sits between "action" and ":").
ACTION_RE = re.compile(r"action[\"']?\s*[:=]\s*(.+)", re.IGNORECASE)
# Words a model may put before a single-character label on its Action line ("Action: Option B", "Action: cell 5").
DESCRIPTOR_RE = re.compile(r"^(?:option|choice|letter|answer|cell|square|position|number|move)\b\s*[:#]?\s*", re.IGNORECASE)
STRIP_CHARS = " \t`'\"*_()[]{}<>.,;:!?"


def strip_think(text: str) -> tuple[str, str]:
    """Remove <think>...</think> blocks; returns (clean_text, think_text)."""
    thinks = THINK_RE.findall(text)
    clean = THINK_RE.sub("", text)
    if "<think>" in clean.lower():  # unterminated block: thinking consumed the budget
        thinks.append(OPEN_THINK_RE.search(clean).group(0) if OPEN_THINK_RE.search(clean) else "")
        clean = OPEN_THINK_RE.sub("", clean)
    elif "</think>" in clean.lower():  # opening tag was in the prompt (templates that pre-fill "<think>")
        m = CLOSE_THINK_RE.search(clean)
        thinks.append(m.group(1))
        clean = clean[m.end():]
    return clean.strip(), "\n".join(t.strip() for t in thinks)


def _match_label(candidate: str, labels: list[str], strict: bool, action_line: bool = False) -> str | None:
    cand = candidate.strip(STRIP_CHARS)
    if not cand:
        return None
    low = {l.lower(): l for l in labels}
    c = cand.lower()
    if c in low:
        return low[c]
    if strict:
        # single-letter labels: accept "A)" / "(a)" / "A." only; on an explicit Action line also "Option B" / "cell 5"
        if action_line:
            c = DESCRIPTOR_RE.sub("", c).strip(STRIP_CHARS)
        head = re.split(r"[\s).:,-]", c, maxsplit=1)[0]
        return low.get(head)
    # prefix / containment on the first token(s)
    for l in sorted(labels, key=len, reverse=True):
        if c.startswith(l.lower()) and (len(c) == len(l) or not c[len(l)].isalnum()):
            return l
    for l in sorted(labels, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(l.lower())}(?![A-Za-z0-9_])", c):
            return l
    if len(cand) >= 4:
        close = difflib.get_close_matches(c, [l.lower() for l in labels], n=1, cutoff=0.8)
        if close:
            return low[close[0]]
    return None


def parse_action(text: str, labels: list[str]) -> tuple[str | None, str]:
    """Return (label or None, think_text). Labels are what the model was asked to emit (names or letters)."""
    clean, think = strip_think(text)
    strict = all(len(l) == 1 for l in labels)
    found = ACTION_RE.findall(clean)
    if found:
        for cand in reversed(found):
            cand = cand.strip().splitlines()[0] if cand.strip() else ""
            m = _match_label(cand, labels, strict, action_line=True)
            if m is not None:
                return m, think
    # No usable "Action:" line: scan lines from the end for a bare label.
    for line in reversed(clean.splitlines()):
        m = _match_label(line, labels, strict)
        if m is not None:
            return m, think
    if not strict:
        # last mention of any label anywhere in the text
        best, best_pos = None, -1
        for l in labels:
            for mm in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(l)}(?![A-Za-z0-9_])", clean, flags=re.IGNORECASE):
                if mm.start() > best_pos:
                    best, best_pos = l, mm.start()
        return best, think
    return None, think
