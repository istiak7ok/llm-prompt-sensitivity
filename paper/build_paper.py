"""Fill table placeholders in paper_src.md and render paper.md (IEEE-numbered refs) and main.tex."""
import json, re, subprocess
from pathlib import Path
import pypandoc

P = Path(__file__).resolve().parent
tabs = json.load(open(P / "_tables.json"))
src = (P / "paper_src.md").read_text()
for k, v in tabs.items():
    src = src.replace("{{" + k + "}}", v)
assert "{{" not in src, re.findall(r"\{\{\w+\}\}", src)
(P / "paper_filled.md").write_text(src)
pd = pypandoc.get_pandoc_path()

# Markdown version with numbered IEEE references
subprocess.run([pd, "paper_filled.md", "--citeproc", "--bibliography=references.bib", "--csl=ieee.csl",
                "-t", "gfm+tex_math_dollars", "--wrap=none", "-s", "-o", "paper.md"], cwd=P, check=True)
md = (P / "paper.md").read_text()
import yaml
fm, body = md.split("---\n", 2)[1:]
meta = yaml.safe_load(fm)
aff = meta.get("affiliation") or "Independent researcher, Bangladesh"
head = (f"# {meta['title']}\n\n**{meta['author']}**  \n*{aff}*  \n*{meta['date']}*\n\n"
        f"> **Disclosure:** drafted with an AI assistant under the author's oversight; see the AI-use disclosure section.\n\n"
        f"## Abstract\n\n{meta['abstract'].strip()}\n\n")
body = re.sub(r"^# ", "## ", body, flags=re.M)
body = body.replace("](figures/", "](figures/")
(P / "paper.md").write_text(head + body.lstrip())

# LaTeX version (natbib, numeric)
subprocess.run([pd, "paper_filled.md", "--natbib", "--bibliography=references.bib", "-s", "-o", "main.tex",
                "-V", "documentclass=article", "-V", "fontsize=10pt", "-V", "geometry:margin=2.2cm",
                "-V", "natbiboptions=numbers,sort&compress", "-V", "biblio-style=unsrtnat",
                "-V", "colorlinks=true",
                "-H", "preamble.tex", "--resource-path=."], cwd=P, check=True)
tex = (P / "main.tex").read_text()
author = meta.get("author", "")
aff = meta.get("affiliation") or "Independent researcher, Bangladesh"
# affiliation under the author name; no email
tex = tex.replace("\\author{" + author + "}",
                  "\\author{" + author + "\\\\" + aff + "}")
tex = tex.replace(".png}", ".pdf}")
tex = tex.replace("Ω", "\\ensuremath{\\Omega}").replace("°", "\\ensuremath{^{\\circ}}")  # pdfLaTeX-safe (arXiv)
tex = re.sub(r"\\textbf\{Table \d+\.\}\s*", "", tex)   # LaTeX numbers tables itself
tex = re.sub(r"\\texttt\{([0-9a-f]{64})\}", r"\\nolinkurl{\1}", tex)   # vector figures in the PDF
# Pandoc wraps longtable headers in \\begin{minipage}[b]{\\linewidth}, which
# makes each header cell a full line wide and overflows the page.
tex = re.sub(
    r"\\begin\{minipage\}\[b\]\{\\linewidth\}(?:\\raggedright|\\raggedleft)?\s*(.*?)\s*\\end\{minipage\}",
    lambda m: m.group(1).strip(),
    tex,
    flags=re.S,
)


def _flatten_p_cols(tex_s):
    """Turn pandoc's p{\real{...}} longtable specs into simple l/r/c columns."""
    tex_s = re.sub(
        r">\{\\(raggedright|raggedleft|centering)\\arraybackslash\}p\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        lambda m: {"raggedright": "l", "raggedleft": "r", "centering": "c"}[m.group(1)],
        tex_s,
    )
    tex_s = re.sub(
        r"\\begin\{longtable\}\[\]\{@\{\}\s*([lrc\s]+)\s*@\{\}\}",
        lambda m: "\\begin{longtable}[]{@{}%s@{}}" % re.sub(r"\s+", "", m.group(1)),
        tex_s,
    )
    return tex_s


def _extract_braced(s, start):
    """Return (inner, index_after_closing_brace) for a '{...}' group at start."""
    if start >= len(s) or s[start] != "{":
        raise ValueError("expected '{' at %d" % start)
    depth = 0
    i = start
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
        i += 1
    raise ValueError("unbalanced brace starting at %d" % start)


def _longtables_to_tables(tex_s):
    """Convert pandoc longtables that fit on one page into floating table environments.

    Captions stay with the tabular body; the table can float so it never
    overflows the text block or the page footer.
    """
    pattern = re.compile(r"\\begin\{longtable\}\[\]\{", re.S)

    def convert_at(s, start):
        # start points at \begin{longtable}[]{
        brace = s.find("{", start + len("\\begin{longtable}[]"))
        colspec, after_spec = _extract_braced(s, brace)
        end_m = re.search(r"\\end\{longtable\}", s[after_spec:])
        if not end_m:
            raise ValueError("unclosed longtable")
        body = s[after_spec:after_spec + end_m.start()].strip()
        end_pos = after_spec + end_m.end()

        cap_at = body.find(r"\caption")
        if cap_at == -1:
            raise ValueError("longtable without caption")
        cap_brace = body.find("{", cap_at)
        caption, after_cap = _extract_braced(body, cap_brace)
        rest = body[after_cap:].lstrip()
        if rest.startswith(r"\tabularnewline"):
            rest = rest[len(r"\tabularnewline"):].lstrip()

        top = re.search(r"\\toprule(?:\\noalign\{\})?", rest)
        mid = re.search(r"\\midrule(?:\\noalign\{\})?", rest)
        if not top or not mid or mid.start() < top.end():
            raise ValueError("could not find header rules in longtable")
        header = rest[top.end():mid.start()].strip()
        header = re.sub(r"(?:\\tabularnewline|\\\\)\s*$", "", header).strip()
        header = header.replace(
            "Q; naive p; cluster-perm. p",
            r"\shortstack[r]{Q; naive p;\\cluster-perm. p}",
        )

        foot = rest.find(r"\endlastfoot")
        if foot == -1:
            raise ValueError("could not find \\endlastfoot")
        rows = rest[foot + len(r"\endlastfoot"):].strip()

        n_cols = len(re.sub(r"@\{\}|[^lrc]", "", colspec))
        if n_cols == 6 and set(re.sub(r"@\{\}", "", colspec)) <= set("l"):
            colspec = r"@{}llll>{\raggedright\arraybackslash}p{0.36\linewidth}l@{}"
            font = r"\footnotesize"
        elif n_cols >= 8:
            font = r"\footnotesize"
        else:
            font = r"\small"

        tabular = (
            "\\begin{table}[tbp]\n"
            "\\centering\n"
            f"{font}\n"
            f"\\caption{{{caption}}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n"
            "\\toprule\n"
            f"{header} \\\\\n"
            "\\midrule\n"
            f"{rows}\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        return tabular, end_pos

    out = []
    pos = 0
    n = 0
    for m in pattern.finditer(tex_s):
        out.append(tex_s[pos:m.start()])
        block, end_pos = convert_at(tex_s, m.start())
        out.append(block)
        pos = end_pos
        n += 1
    out.append(tex_s[pos:])
    print(f"converted {n} longtables to table floats")
    return "".join(out)


tex = _flatten_p_cols(tex)
tex = _longtables_to_tables(tex)
# Keep figures from requesting a full text-height box (overflows the page once a caption is added).
tex = tex.replace(
    "width=0.85\\linewidth,height=\\textheight,keepaspectratio",
    "width=0.85\\linewidth,height=0.42\\textheight,keepaspectratio",
)
tex = tex.replace(
    "width=0.95\\linewidth,height=\\textheight,keepaspectratio",
    "width=0.95\\linewidth,height=0.42\\textheight,keepaspectratio",
)

(P / "main.tex").write_text(tex)
print("ok")
