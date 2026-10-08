#!/usr/bin/env python3
"""Build PDFs of the paper draft and the project handbook without a TeX installation.

paper:    LaTeX sections -> (expand inputs, resolve cross-references and missing assets) -> pandoc with
          citeproc (Harvard "Cite Them Right" style) and KaTeX -> Chromium print to PDF.
handbook: the Markdown plan and guides -> pandoc -> Chromium.

Usage: python scripts/build_draft_pdf.py [--paper] [--handbook]   (both by default)
Requires: pandoc, the katex npm package under build/node_modules (cd build && npm install katex), the Harvard
CSL file in paper/, and Playwright with a Chromium binary (PLAYWRIGHT_BROWSERS_PATH or /opt/pw-browsers).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER, BUILD = ROOT / "paper", ROOT / "build"
CSL = PAPER / "harvard-cite-them-right.csl"

MACROS = r"""
\newcommand{\deadeye}{\textsc{DeadEye}}
\newcommand{\norm}{\ensuremath{s}}
\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}
\newcommand{\result}[1]{\textcolor{blue}{[result pending: #1]}}
"""

CSS = """
@page { size: A4; margin: 22mm 20mm 24mm 20mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.42; color: #111; max-width: 100%; }
h1.title { font-size: 20pt; line-height: 1.25; margin-bottom: 4pt; }
p.author, p.date, p.subtitle { margin: 2pt 0; color: #333; }
h1 { font-size: 15pt; margin-top: 22pt; } h2 { font-size: 12.5pt; margin-top: 16pt; } h3 { font-size: 11pt; }
h4 { font-size: 10.5pt; display: inline; margin-right: 6pt; } h4 + p { display: inline; }
table { border-collapse: collapse; font-size: 8.5pt; margin: 8pt auto; width: 100%; }
th, td { border-bottom: 1px solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }
thead th { border-bottom: 1.5px solid #333; } caption { caption-side: top; font-size: 9pt; text-align: left; margin-bottom: 4pt; }
pre { font-size: 7.6pt; white-space: pre-wrap; background: #f6f6f4; padding: 6pt; border: 1px solid #ddd; }
code { font-size: 8.8pt; } .draft-banner { border: 2px solid #b00; color: #b00; padding: 6pt 8pt; margin: 10pt 0 14pt; font-size: 10pt; }
#refs { font-size: 9.2pt; } .csl-entry { margin-bottom: 4pt; padding-left: 1.5em; text-indent: -1.5em; }
figure { text-align: center; margin: 10pt 0; } figcaption { font-size: 9pt; text-align: left; }
.katex { font-size: 1.02em; } nav#TOC { font-size: 9pt; columns: 2; }
"""


def strip_comments(tex: str) -> str:
    """Drop LaTeX comments outside verbatim blocks (pandoc copes, but the helpers below parse raw text)."""
    parts = re.split(r"(\\begin\{verbatim\}.*?\\end\{verbatim\})", tex, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?<!\\)%.*", "", parts[i])
    return "".join(parts)


def expand_inputs(tex: str, base: Path) -> str:
    def rep(m):
        name = m.group(1)
        target = base / (name if name.endswith(".tex") else name + ".tex")
        return expand_inputs(target.read_text(), base) if target.exists() else f"\\textcolor{{red}}{{[missing input {name}]}}"
    return re.sub(r"\\input\{([^}]*)\}", rep, tex)


def _balanced(tex: str, k: int) -> tuple[str, int]:
    """Return the content of the brace group starting at tex[k] == '{' and the index after it."""
    assert tex[k] == "{", tex[k:k + 30]
    depth, start = 0, k
    while True:
        c = tex[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return tex[start + 1:k], k + 1
        k += 1


def resolve_iffileexists(tex: str) -> str:
    out, i, key = [], 0, "\\IfFileExists"
    while True:
        j = tex.find(key, i)
        if j < 0:
            out.append(tex[i:])
            return "".join(out)
        out.append(tex[i:j])
        k = j + len(key)
        args = []
        for _ in range(3):
            a, k = _balanced(tex, k)
            args.append(a)
        fname, then, els = args
        out.append(then if (PAPER / fname).exists() else els)
        i = k


def strip_wrappers(tex: str) -> str:
    """Remove \\adjustbox wrappers, turn missing-asset notes into placeholders, simplify tabularx and floats."""
    out, i, key = [], 0, "\\adjustbox"
    while True:
        j = tex.find(key, i)
        if j < 0:
            out.append(tex[i:])
            break
        out.append(tex[i:j])
        _, k = _balanced(tex, j + len(key))  # the options group
        body, k2 = _balanced(tex, k)
        out.append(body)
        i = k2
    tex = "".join(out)
    tex = re.sub(r"\\fbox\{[^}]*missing[^}]*\}", r"\\textcolor{red}{[Figure placeholder: generated from the results by deadeye report]}", tex)
    tex = re.sub(r"\((tables/[^)]*missing[^)]*)\)", r"\\textcolor{red}{[Table placeholder: generated from the results by deadeye report]}", tex)
    tex = re.sub(r"\\begin\{tabularx\}\{\\linewidth\}\{([^}]*)\}", lambda m: "\\begin{tabular}{" + m.group(1).replace("@{}", "").replace("X", "l") + "}", tex)
    tex = tex.replace("\\end{tabularx}", "\\end{tabular}")
    tex = re.sub(r"\\begin\{(table|figure)\}\[[^\]]*\]", r"\\begin{\1}", tex)
    tex = tex.replace("\\centering\\small", "").replace("\\centering", "").replace("\\small", "")
    tex = re.sub(r"\\(begin|end)\{footnotesize\}", "", tex)
    return tex


def resolve_refs(tex: str) -> str:
    """Number sections, equations, tables and figures in order and replace \\ref/\\eqref with the numbers."""
    labels: dict[str, str] = {}
    sec = [0, 0, 0]
    appendix = False
    eq = tab = fig = 0
    pending: str | None = None
    tokens = re.compile(r"\\(section|subsection|subsubsection|appendix|label|begin\{equation\}|begin\{table\}|begin\{figure\})(\*?)(?:\{([^}]*)\})?")
    for m in tokens.finditer(tex):
        kind, arg = m.group(1), m.group(3)
        if kind == "appendix":
            appendix, sec[:] = True, [0, 0, 0]
        elif kind in ("section", "subsection", "subsubsection") and not m.group(2):
            lvl = ("section", "subsection", "subsubsection").index(kind)
            sec[lvl] += 1
            for i in range(lvl + 1, 3):
                sec[i] = 0
            num = ".".join(str(x) for x in sec[:lvl + 1])
            if appendix:
                num = chr(ord("A") + sec[0] - 1) + ("." + ".".join(str(x) for x in sec[1:lvl + 1]) if lvl else "")
            pending = num
        elif kind == "begin{equation}":
            eq += 1
            pending = str(eq)
        elif kind == "begin{table}":
            tab += 1
            pending = str(tab)
        elif kind == "begin{figure}":
            fig += 1
            pending = str(fig)
        elif kind == "label" and arg and pending is not None:
            labels.setdefault(arg, pending)
    tex = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({labels.get(m.group(1), '?')})", tex)
    tex = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), tex)
    return tex


def build_paper() -> Path:
    main = (PAPER / "main.tex").read_text()
    title = re.search(r"\\title\{(.*?)\}\s*\n\\author", main, flags=re.S).group(1)
    title = re.sub(r"\\\\\s*", " ", title).replace("\\large", "").strip()
    body = main[main.index("\\begin{document}") + len("\\begin{document}"):main.index("\\end{document}")]
    body = body.replace("\\maketitle", "").replace("\\bibliographystyle{agsm}", "").replace("\\bibliography{refs}", "")
    body = strip_comments(expand_inputs(body, PAPER))
    body = resolve_iffileexists(body)
    body = strip_wrappers(body)
    body = resolve_refs(body)
    # KaTeX has no \label or \ensuremath: drop labels inside equations (numbers are already resolved).
    body = re.sub(r"(\\begin\{equation\}.*?\\end\{equation\})", lambda m: re.sub(r"\\label\{[^}]*\}", "", m.group(1)), body, flags=re.S)
    banner = ("\\textbf{Draft.} This is the complete draft of the paper with the empirical results still to be measured: "
              "every blue ``result pending'' mark is a number the main sweep will fill in, and every red TODO is an open item. "
              "Figures and tables that read ``placeholder'' are generated from the results by the report tool.")
    doc = (MACROS + f"\\title{{{title}}}\n\\author{{DeadEye project}}\n\\date{{Draft built 8 October 2026}}\n"
           f"\\begin{{document}}\n{banner}\n\n{body}\n\\end{{document}}\n")
    BUILD.mkdir(exist_ok=True)
    (BUILD / "paper_flat.tex").write_text(doc)
    (BUILD / "style.css").write_text(CSS)
    html = BUILD / "paper.html"
    cmd = ["pandoc", str(BUILD / "paper_flat.tex"), "-f", "latex", "-t", "html5", "-s", "--katex=katex/",
           "--citeproc", "--bibliography", str(PAPER / "refs.bib"), "--csl", str(CSL), "--number-sections",
           "--css", "style.css", "-M", "reference-section-title=References", "-M", "link-citations=true", "-o", str(html)]
    subprocess.run(cmd, check=True)
    html.write_text(html.read_text().replace("\\ensuremath{s}", "s"))
    katex_src = BUILD / "node_modules" / "katex" / "dist"
    if katex_src.exists():
        shutil.copytree(katex_src, BUILD / "katex", dirs_exist_ok=True)
    return html_to_pdf(html, BUILD / "DeadEye_paper_draft.pdf")


def build_handbook() -> Path:
    parts = [("Before you publish", ROOT / "docs" / "before_you_publish.md"), ("Research plan", ROOT / "RESEARCH_PLAN.md"),
             ("Pre-registration", ROOT / "docs" / "preregistration.md"), ("Testing as a peer", ROOT / "docs" / "peer_testing.md"),
             ("Compute budget", ROOT / "docs" / "compute_budget.md"), ("Known limitations", ROOT / "docs" / "known_limitations.md"),
             ("References audit", ROOT / "docs" / "references_audit.md")]
    md = ["---", "title: DeadEye handbook",
          "subtitle: pre-publication guide, research plan, pre-registration, peer testing, budget, limitations, references audit",
          "date: 8 October 2026", "---", ""]
    for name, path in parts:
        text = path.read_text()
        text = re.sub(r"^# .*\n", "", text, count=1)  # drop the file's own top title
        text = re.sub(r"^(#+) ", lambda m: "#" + m.group(1) + " ", text, flags=re.M)  # demote headings one level
        md.append(f"# {name}\n\n{text}\n\n")
    BUILD.mkdir(exist_ok=True)
    (BUILD / "style.css").write_text(CSS)
    src = BUILD / "handbook.md"
    src.write_text("\n".join(md))
    html = BUILD / "handbook.html"
    subprocess.run(["pandoc", str(src), "-f", "gfm+yaml_metadata_block", "-t", "html5", "-s", "--toc", "--toc-depth=2",
                    "--css", "style.css", "-o", str(html)], check=True)
    return html_to_pdf(html, BUILD / "DeadEye_handbook.pdf")


def html_to_pdf(html: Path, pdf: Path) -> Path:
    from playwright.sync_api import sync_playwright

    exe = next((c for c in ("/opt/pw-browsers/chromium", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome") if Path(c).is_file()), None)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = browser.new_page()
        page.goto(html.resolve().as_uri(), wait_until="load")
        page.wait_for_timeout(1500)
        page.pdf(path=str(pdf), format="A4", print_background=True,
                 margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}, display_header_footer=True,
                 header_template="<span></span>",
                 footer_template="<div style='font-size:7pt;color:#666;width:100%;text-align:center;'><span class='pageNumber'></span></div>")
        browser.close()
    return pdf


if __name__ == "__main__":
    which = sys.argv[1:] or ["--paper", "--handbook"]
    if "--paper" in which:
        print("paper ->", build_paper())
    if "--handbook" in which:
        print("handbook ->", build_handbook())
