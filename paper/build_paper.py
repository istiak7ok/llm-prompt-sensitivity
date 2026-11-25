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



def _split_cells(row: str):
    cells, buf, depth, i = [], [], 0, 0
    while i < len(row):
        if row[i] == "\\":
            buf.append(row[i:i + 2])
            i += 2
            continue
        ch = row[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "&" and depth == 0:
            cells.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    cells.append("".join(buf).strip())
    return cells


def _split_rows(body: str):
    """Split a tabular body into rows on top-level \\\\."""
    body = body.strip()
    if body.endswith(r"\\"):
        body = body[:-2].rstrip()
    rows, buf, depth, i = [], [], 0, 0
    while i < len(body):
        if body[i] == "\\":
            # row break \\ but not \tabularnewline already converted
            if i + 1 < len(body) and body[i + 1] == "\\" and depth == 0:
                # not \\\\ inside a later macro we inserted (none yet)
                nxt = body[i + 2:i + 3]
                rows.append("".join(buf).strip())
                buf = []
                i += 2
                continue
            buf.append(body[i:i + 2])
            i += 2
            continue
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
        buf.append(body[i])
        i += 1
    tail = "".join(buf).strip()
    if tail:
        rows.append(tail)
    return rows


def _stack_ci(cell: str) -> str:
    s = re.sub(r"\s+", " ", cell).strip()
    m = re.fullmatch(
        r"(\d+\.\d+)\s*\{\[\}(\d+\.\d+),\s*(\d+\.\d+)\{\]\}", s
    )
    if not m:
        return cell
    a, b, c = m.groups()
    return r"\makecell[r]{%s\\{[%s, %s]}}" % (a, b, c)


def _stack_slash(cell: str) -> str:
    s = re.sub(r"\s+", " ", cell).strip()
    m = re.fullmatch(r"(\d+\.\d+)\s*/\s*(\d+(?:\.\d+)?)", s)
    if not m:
        return cell
    return r"\makecell[r]{%s\\%s}" % m.groups()


def _stack_paren(cell: str) -> str:
    s = re.sub(r"\s+", " ", cell).strip()
    m = re.fullmatch(r"(\d+\.\d+)\s*\(([^)]+)\)", s)
    if not m:
        return cell
    return r"\makecell[r]{%s\\(%s)}" % m.groups()


def _stack_qtriple(cell: str) -> str:
    s = re.sub(r"\s+", " ", cell).strip()
    m = re.fullmatch(r"(\d+\.\d+);\s*(\d+\.\d+);\s*(\d+\.\d+)", s)
    if not m:
        return cell
    a, b, c = m.groups()
    return r"\makecell[r]{%s\\%s; %s}" % (a, b, c)


def _join_rows(rows_cells, space_when=None):
    """rows_cells is a list of list-of-cells. space_when(prev, cur) -> bool."""
    out = []
    prev = None
    for i, cells in enumerate(rows_cells):
        line = " & ".join(cells) + r" \\"
        out.append(line)
        if i < len(rows_cells) - 1:
            nxt = rows_cells[i + 1]
            if space_when is not None and space_when(cells, nxt):
                out.append(r"\addlinespace")
        prev = cells
    return "\n".join(out)


def _model_changed(prev, cur):
    return prev and cur and prev[0] != cur[0]


def _polish_one_table(block: str) -> str:
    cap_at = block.find(r"\caption")
    brace = block.find("{", cap_at)
    caption, after = _extract_braced(block, brace)
    cap_l = re.sub(r"\s+", " ", caption)
    font_m = re.search(r"\\(small|footnotesize)\b", block)
    font = r"\\" + (font_m.group(1) if font_m else "small")
    tab_at = block.find(r"\begin{tabular}")
    if tab_at == -1:
        return block
    spec_brace = block.find("{", tab_at + len(r"\begin{tabular}"))
    colspec, after_spec = _extract_braced(block, spec_brace)
    inner = block[after_spec:]
    inner = re.sub(r"^.*?\\toprule\s*", "", inner, count=1, flags=re.S)
    mid = inner.find(r"\midrule")
    bot = inner.rfind(r"\bottomrule")
    header = inner[:mid].strip()
    header = re.sub(r"(?:\\tabularnewline|\\\\)\s*$", "", header).strip()
    rows_tex = inner[mid + len(r"\midrule"):bot].strip()
    row_strs = _split_rows(rows_tex)
    rows = [_split_cells(r) for r in row_strs]

    note = ""
    space_when = _model_changed

    if "Model-level results over all four styles" in cap_l:
        font = r"\small"
        header = r"Model & Acc. & Hall. & CR & Surf. & PS & QS & Q (p)"
        new = []
        for cells in rows:
            cells = list(cells)
            # stack CI in accuracy, halluc, CR, surface, PS (indices 1-5)
            for j in range(1, min(6, len(cells))):
                cells[j] = _stack_ci(cells[j])
            if len(cells) >= 8:
                cells[7] = _stack_paren(cells[7])
            new.append(cells)
        rows = new
        note = (
            r"\par\nopagebreak\vspace{2pt}{\footnotesize\textit{Note.} "
            r"Acc.\ = accuracy (\%); Hall.\ = hallucination rate (\%); "
            r"CR = label consistency; Surf.\ = surface consistency; "
            r"PS = accuracy spread (percentage points); "
            r"QS = \% of questions whose correctness changes; "
            r"Q (p) = Cochran's Q (p-value). "
            r"Brackets are 95\% bootstrap CIs.}"
        )

    elif "Pooled over the four models" in cap_l:
        font = r"\small"
        header = (
            r"Metric & Simple & Detailed & Polite & CoT & "
            r"\makecell[r]{Q; naive $p$;\\cluster $p$}"
        )
        new = []
        for cells in rows:
            cells = list(cells)
            for j in range(1, min(5, len(cells))):
                cells[j] = _stack_ci(cells[j])
            if len(cells) >= 6:
                cells[5] = _stack_qtriple(cells[5])
            new.append(cells)
        rows = new
        space_when = lambda a, b: True
        note = (
            r"\par\nopagebreak\vspace{2pt}{\footnotesize\textit{Note.} "
            r"Each style column is the pooled rate with a question-clustered "
            r"95\% bootstrap CI underneath. The last column is Cochran's Q, "
            r"the naive $p$-value, and the question-clustered permutation $p$-value.}"
        )

    elif "Share of final answers longer than five words" in cap_l:
        font = r"\small"
        header = (
            r"Model & \makecell{Simple\\(\% / tok.)} & "
            r"\makecell{Detailed\\(\% / tok.)} & "
            r"\makecell{Polite\\(\% / tok.)} & "
            r"\makecell{CoT\\(\% / tok.)} & "
            r"\makecell{Cochran\\Q ($p$)}"
        )
        new = []
        for cells in rows:
            cells = list(cells)
            for j in range(1, min(5, len(cells))):
                cells[j] = _stack_slash(cells[j])
            if len(cells) >= 6:
                cells[5] = _stack_paren(cells[5])
            new.append(cells)
        rows = new
        note = (
            r"\par\nopagebreak\vspace{2pt}{\footnotesize\textit{Note.} "
            r"In each style cell the upper value is the share of final answers "
            r"longer than five words (\%) and the lower value is the median "
            r"output-token count. Medians of an even number of integer counts "
            r"can end in .5.}"
        )

    elif "Accuracy with 95" in cap_l and "per model and style" in cap_l:
        font = r"\small"
        header = r"Model & Style & Accuracy [95\% CI] & Hall. & Format"
        # Leave CIs on one line so the table can sit with the Results intro.

    elif "alternative grading choices" in cap_l:
        font = r"\small"
        header = (
            r"Model & v1 & v2 & Excl.\ K09/S22 & No FA & N-S wrong"
        )
        note = (
            r"\par\nopagebreak\vspace{2pt}{\footnotesize\textit{Note.} "
            r"v1 = strict first grader; v2 = final grader; "
            r"Excl.\ K09/S22 = those two items dropped; "
            r"No FA = rows without a ``Final answer'' line counted wrong; "
            r"N-S wrong = ``Nur-Sultan'' not accepted as the capital of Kazakhstan.}"
        )

    elif "Pass 1 vs pass 2" in cap_l:
        font = r"\small"
        header = r"Model & Pairs & Raw & Extracted & Label"
        note = (
            r"\par\nopagebreak\vspace{2pt}{\footnotesize\textit{Note.} "
            r"Raw = identical raw response text; Extracted = identical extracted "
            r"answer; Label = identical correctness label. Values are \% of 120 pairs, "
            r"except Pairs, which is the count.}"
        )

    elif "All non-correct answers" in cap_l:
        font = r"\small"
        colspec = r"@{}llll>{\raggedright\arraybackslash}p{0.38\linewidth}l@{}"
        header = r"Model & Item & Style & Gold & Extracted answer (abbrev.) & Code"

    body = _join_rows(rows, space_when=space_when)
    out = (
        "\\begin{table}[htb]\n"
        "\\centering\n"
        f"{font}\n"
        f"\\caption{{{caption}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"{note}\n"
        "\\end{table}"
    )
    return out


def _polish_tables(tex_s: str) -> str:
    pat = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.S)
    n = [0]

    def repl(m):
        n[0] += 1
        return _polish_one_table(m.group(0))

    out = pat.sub(repl, tex_s)
    print(f"polished {n[0]} tables")
    return out


tex = _flatten_p_cols(tex)
tex = _longtables_to_tables(tex)
tex = _polish_tables(tex)
# Keep figures compact enough to sit with text (avoid a half-empty float page).
tex = tex.replace(
    "width=0.85\\linewidth,height=\\textheight,keepaspectratio",
    "width=0.88\\linewidth,height=0.30\\textheight,keepaspectratio",
)
tex = tex.replace(
    "width=0.95\\linewidth,height=\\textheight,keepaspectratio",
    "width=0.88\\linewidth,height=0.30\\textheight,keepaspectratio",
)
# Also rewrite any already-capped figure includes from a previous pass's leftover
# (build always starts from pandoc, which uses \textheight).
tex = tex.replace(
    "width=0.85\\linewidth,height=0.42\\textheight,keepaspectratio",
    "width=0.88\\linewidth,height=0.30\\textheight,keepaspectratio",
)
tex = tex.replace(
    "width=0.95\\linewidth,height=0.42\\textheight,keepaspectratio",
    "width=0.88\\linewidth,height=0.30\\textheight,keepaspectratio",
)

tex = tex.replace("\\end{figure}", "\\end{figure}\n\\FloatBarrier")

(P / "main.tex").write_text(tex)
print("ok")
