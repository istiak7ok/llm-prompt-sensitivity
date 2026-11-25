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

tex = _flatten_p_cols(tex)
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
