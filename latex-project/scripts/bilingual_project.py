"""Generate book and standalone TeX from the same language-specific sections.

No translation is performed here. Only reviewed Markdown is converted.
Explicit raw-LaTeX unit markers, not inferred headings, define split boundaries.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "latex-project"
EXPECTED_SECTIONS = 168
PART_NAMES = {
    "01-deep-learning": "Deep Learning",
    "02-reinforcement-learning": "Reinforcement Learning",
    "03-large-language-model": "Large Language Models",
    "04-llm-agents": "LLM Agents",
    "05-diffusion-multimodal-generation": "Diffusion Models and Multimodal Generation",
    "06-embodied-ai-world-models": "Embodied AI and World Models",
}
EN_ROOT = ["README_EN.md", "BEGINNER_LEARNING_PATH_EN.md", "DISCLAIMER_EN.md",
           "CONTRIBUTING_EN.md", "CHANGELOG_EN.md"]
MARKER = re.compile(r"^% AINOTE-UNIT ([a-z0-9-]+)$", re.M)


@dataclass(frozen=True)
class Section:
    id: str
    language: str
    markdown: str
    title: str
    chapter_title: str
    chapter_directory: str
    part: str
    part_title: str
    tex: str
    standalone: str


def write(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def discover(engine: ModuleType, language: str) -> list[Section]:
    folder = ROOT / ("docs" if language == "zh" else "docs-en")
    sections = []
    ids = set()
    for path in sorted(folder.rglob("*.md")):
        match = re.match(r"^(\d{2}-\d{2})-", path.name)
        if not match:
            raise ValueError(f"Invalid section filename: {path}")
        section_id = match[1]
        if section_id in ids:
            raise ValueError(f"Duplicate {language} section: {section_id}")
        ids.add(section_id)
        front, body = engine.split_front_matter(engine.read_text(path))
        meta = dict(engine.parse_front_matter(front))
        title = engine.first_heading_title(body, meta.get("title", path.stem))
        title = engine.strip_leading_number(title)
        relative = path.relative_to(ROOT)
        part = relative.parts[1]
        if part not in PART_NAMES or len(relative.parts) != 4:
            raise ValueError(f"Invalid section hierarchy: {relative}")
        chapter = meta.get("chapter_title") or engine.strip_leading_number(path.parent.name)
        if language == "en" and not meta.get("chapter_title"):
            raise ValueError(f"English chapter_title metadata is required: {relative}")
        part_title = engine.PART_TITLES[part] if language == "zh" else f"Part {int(part[:2])} {PART_NAMES[part]}"
        sections.append(Section(section_id, language, relative.as_posix(), title, chapter,
                                path.parent.name, part, part_title,
                                f"content/{language}/sections/{section_id}.tex",
                                f"standalone/{language}/{section_id}.tex"))
    return sections


class StableImageRegistry:
    """Keep old curated image mappings stable; never remove an image tree."""

    def __init__(self) -> None:
        self.path = PROJECT / "resource-map.json"
        if self.path.exists():
            self.mapping = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            manifest = PROJECT / "RESOURCE_MANIFEST.md"
            text = manifest.read_text(encoding="utf-8-sig") if manifest.exists() else ""
            self.mapping = dict(re.findall(r"\| `([^`]+)` \| `(images/[^`]+)` \|", text))
        self.used: dict[str, str] = {}

    def register(self, source: Path) -> str:
        source = source.resolve()
        source.relative_to((ROOT / "assets" / "images").resolve())
        key = source.relative_to(ROOT).as_posix()
        if key not in self.mapping:
            numbers = [int(m[1]) for value in self.mapping.values()
                       if (m := re.search(r"image-(\d+)", value))]
            suffix = ".png" if source.suffix.lower() == ".webp" else source.suffix.lower()
            self.mapping[key] = f"images/image-{max(numbers, default=0) + 1:04d}{suffix}"
        relative = self.mapping[key]
        destination = (PROJECT / relative).resolve()
        destination.relative_to((PROJECT / "images").resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".webp":
            from PIL import Image
            with Image.open(source) as visual:
                converted = BytesIO()
                visual.save(converted, "PNG")
            encoded = converted.getvalue()
            if not destination.exists() or destination.read_bytes() != encoded:
                destination.write_bytes(encoded)
        elif not destination.exists() or source.read_bytes() != destination.read_bytes():
            # Byte equality only avoids needless copies; no hashes are computed.
            shutil.copy2(source, destination)
        self.used[key] = relative
        return relative

    def save(self) -> None:
        write(self.path, json.dumps(self.mapping, ensure_ascii=False, indent=2) + "\n")


def protect_layout_boundaries(content: str) -> str:
    """Keep structural headings and short colon-ended introductions with content."""
    literal = re.compile(
        r"(?ms)^[ \t]*\\begin\{(verbatim|Shaded)\}[ \t]*$.*?^[ \t]*\\end\{\1\}[ \t]*$")

    def guard(prose: str) -> str:
        prose = re.sub(
            r"(?m)(?<!\\Needspace\{5\\baselineskip\}\n)^(?=\\hypertarget\{[^{}\n]+\}\{%\n\\(?:section|subsection|subsubsection)\*?\{)",
            lambda _: "\\Needspace{5\\baselineskip}\n", prose)
        prose = re.sub(
            r"(?m)(?<!\\Needspace\{8\\baselineskip\}\n)^(（[0-9]+）[^\n.!?。！？]{1,120})$",
            lambda match: "\\Needspace{8\\baselineskip}\n" + match.group(1), prose)
        prose = re.sub(
            r"(?mi)(?<!\\Needspace\{8\\baselineskip\}\n)^((?:[0-9]+\.\s+|Step\s+[0-9]+[：:]\s*|步骤\s*[0-9]+\s*[：:]\s*)[^\n]{1,120})$",
            lambda match: "\\Needspace{8\\baselineskip}\n" + match.group(1), prose)
        prose = re.sub(
            r"(?m)(?<!\\Needspace\{4\\baselineskip\}\n)^(?=\\begin\{enumerate\}\n(?:\\(?:def|setcounter|tightlist)[^\n]*\n)*\\item\n  [^\n.!?。！？]{1,120}\n\\end\{enumerate\})",
            lambda _: "\\Needspace{4\\baselineskip}\n", prose)
        return re.sub(
            r"(?m)(?<!\\Needspace\{5\\baselineskip\}\n)^([^\n]{1,240}[：:])$",
            lambda match: "\\Needspace{5\\baselineskip}\n" + match.group(1), prose)

    chunks: list[str] = []
    cursor = 0
    for match in literal.finditer(content):
        chunks.extend((guard(content[cursor:match.start()]), match.group(0)))
        cursor = match.end()
    chunks.append(guard(content[cursor:]))
    return "".join(chunks)


def protect_short_verbatim(content: str) -> str:
    """Keep small literal code examples intact when they fit on one page."""
    pattern = re.compile(
        r"(?ms)(?<!\\Needspace\{10\\baselineskip\}\n)^(\\begin\{verbatim\}\n.*?^\\end\{verbatim\})$")

    def guard(match: re.Match[str]) -> str:
        block = match.group(1)
        if block.count("\n") > 8:
            return block
        return "\\Needspace{10\\baselineskip}\n" + block

    return pattern.sub(guard, content)


def repaired_tex(engine: ModuleType, text: str) -> str:
    text = re.sub(r"\\text\{([^{}]*[\u4e00-\u9fff][^{}]*)\}", r"\\mathcjk{\1}", text)
    text = re.sub(r"\\mbox\{([^{}]*[\u4e00-\u9fff][^{}]*)\}", r"\\mathcjk{\1}", text)
    text = engine.wrap_cjk_inside_math(text)
    text = text.replace("\\begin{figure}\n", "\\begin{figure}[htbp]\n")
    text = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}([lcr]+)@\{\}\}",
                  engine.expand_simple_longtable_columns, text)
    text = engine.normalize_fractional_longtable_widths(text)
    text = engine.repair_generated_math_notation(text)
    text = engine.patch_wide_display_content(text)
    # English arrow labels need more room than the Chinese labels. Preserve the
    # complete sequence and split only its display layout into three lines.
    policy_sequence = r"\pi^{0} \xrightarrow{\text{policy evaluation}} V^{\pi^{0}} \xrightarrow{\text{policy improvement}} \pi^{1} \xrightarrow{\text{policy evaluation}} V^{\pi^{1}} \xrightarrow{\text{policy improvement}} \pi^{2} \xrightarrow{\text{policy evaluation}} \cdots \xrightarrow{\text{policy improvement}} \pi^{*}"
    policy_reflow = r"""\begin{aligned}
\pi^{0} &\xrightarrow{\text{policy evaluation}} V^{\pi^{0}} \xrightarrow{\text{policy improvement}} \pi^{1} \\
&\xrightarrow{\text{policy evaluation}} V^{\pi^{1}} \xrightarrow{\text{policy improvement}} \pi^{2} \\
&\xrightarrow{\text{policy evaluation}} \cdots \xrightarrow{\text{policy improvement}} \pi^{*}
\end{aligned}"""
    text = text.replace(policy_sequence, policy_reflow)
    # Keep the original NF4 example values; a display avoids overlong English
    # prose plus an indivisible inline vector in nested lists.
    for vector in ("[0.214, -0.419, 0.733, -0.057, 0.024, 0.471, -0.157, -1.0]",
                   "[0.1906, -0.3949, 0.6717, -0.0911, 0.0062, 0.4045, -0.1848, -1.0]"):
        text = text.replace(r"\(" + vector + r"\)", "\n\n\\[\n" + vector + "\n\\]")
    for image_name in ("image-0014.jpeg", "image-0069.png"):
        text = text.replace(r"\includegraphics{images/" + image_name + "}",
                            r"\includegraphics[width=0.82\linewidth]{images/" + image_name + "}")
    text = re.sub(r"(\\subsection\{(?:参考文献|References)\}\\label\{[^{}\n]+\}\})\n\n(?=\\begin\{itemize\})",
                  r"\1\n\n\\vspace{0.25em}\n", text)
    text = protect_layout_boundaries(text)
    text = re.sub(
        r"(?m)(?<!\\FloatBarrier\n)(?<!\\Needspace\{5\\baselineskip\}\n)^(?=(?:\\Needspace\{5\\baselineskip\}\n)?\\hypertarget\{[^{}\n]+\}\{%\n\\subsection\{(?:参考文献|References)\})",
        lambda _: "\\FloatBarrier\n", text)
    text = protect_short_verbatim(text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def split_units(text: str, expected: list[tuple[str, str]]) -> dict[str, str]:
    markers = list(MARKER.finditer(text))
    if [m[1] for m in markers] != [token for token, _ in expected]:
        raise ValueError("Pandoc unit markers are missing, duplicated or reordered")
    if markers and text[:markers[0].start()].strip():
        raise ValueError("Unassigned TeX before first unit marker")
    return {path: text[m.end():markers[i+1].start() if i+1 < len(markers) else len(text)].strip() + "\n"
            for i, (m, (_, path)) in enumerate(zip(markers, expected, strict=True))}


def protect_frontmatter_headings(content: str) -> str:
    # Reserve room before the hyperlink target as well as the heading so a
    # page break never leaves the navigation anchor on the previous page.
    return protect_layout_boundaries(content)


def generate_language(engine: ModuleType, language: str, sections: list[Section],
                      registry: StableImageRegistry, allow_partial: bool) -> None:
    units: list[tuple[str, str]] = []
    chunks: list[str] = []

    def add(token: str, relative: str, body: str) -> None:
        units.append((token, relative))
        chunks.extend([engine.raw_latex(f"% AINOTE-UNIT {token}"), body, ""])

    prefix = f"content/{language}"
    front_setup = "\n".join([r"\setcounter{secnumdepth}{-1}", r"\setcounter{tocdepth}{0}",
                              r"\addtocontents{toc}{\protect\setcounter{tocdepth}{0}}"])
    add("front-setup", f"{prefix}/frontmatter/000-setup.tex", engine.raw_latex(front_setup))
    root_files = engine.ROOT_MARKDOWN if language == "zh" else EN_ROOT
    for index, name in enumerate(root_files, 1):
        path = ROOT / name
        if not path.exists():
            if allow_partial:
                continue
            raise FileNotFoundError(f"Missing {language} front matter: {name}")
        text = engine.read_text(path)
        # UI language controls are navigation, not book content.
        text = re.sub(r"<!-- language-switch:start -->.*?<!-- language-switch:end -->", "", text, flags=re.S)
        text = engine.normalize_markdown_for_latex(engine.rewrite_images(text, path, registry))
        add(f"front-{index:03d}", f"{prefix}/frontmatter/{index:03d}.tex", text + "\n\n" + engine.raw_latex(r"\clearpage"))
    setup = "\n".join([r"\mainmatter", r"\setcounter{secnumdepth}{1}", r"\setcounter{tocdepth}{1}",
                        r"\addtocontents{toc}{\protect\setcounter{tocdepth}{1}}"])
    add("main-setup", f"{prefix}/chapters/000-setup.tex", engine.raw_latex(setup))
    previous_part = previous_chapter = ""
    for section in sections:
        chapter = section.id[:2]
        if section.part != previous_part:
            add(f"part-{section.part[:2]}", f"{prefix}/parts/{section.part[:2]}.tex", engine.part_heading(section.part_title))
            previous_part = section.part
        if chapter != previous_chapter:
            # Explicit counter preserves numbering in partial developer builds.
            text = engine.raw_latex(rf"\setcounter{{chapter}}{{{int(chapter)-1}}}") + f"\n\n# {section.chapter_title}\n"
            add(f"chapter-{chapter}", f"{prefix}/chapters/{chapter}.tex", text)
            previous_chapter = chapter
        path = ROOT / section.markdown
        front, body = engine.split_front_matter(engine.read_text(path))
        body = engine.rewrite_images(body, path, registry)
        body = engine.normalize_markdown_for_latex(engine.section_markdown(body, front, path.stem))
        body = engine.raw_latex(rf"\setcounter{{section}}{{{int(section.id[3:])-1}}}") + "\n\n" + body
        add(f"section-{section.id}", section.tex, body + "\n\n" + engine.raw_latex(r"\clearpage"))
    work = PROJECT / "build" / "generated" / language
    combined, converted = work / "combined.md", work / "combined.tex"
    write(combined, "\n\n".join(chunks))
    subprocess.run(["pandoc", str(combined), "--from", "markdown+tex_math_dollars+raw_attribute+pipe_tables+fenced_code_blocks",
                    "--to", "latex", "--top-level-division=chapter", "--wrap=preserve", "--no-highlight",
                    "--resource-path", str(PROJECT), "-o", str(converted)], cwd=PROJECT, check=True)
    tex = repaired_tex(engine, converted.read_text(encoding="utf-8"))
    output = split_units(tex, units)
    for relative, content in output.items():
        if "/frontmatter/" in relative:
            # A heading followed by a list can otherwise remain alone at the
            # bottom of a page under the existing front-matter title style.
            content = protect_frontmatter_headings(content)
        write(PROJECT / relative, content)
    body_name = "body.tex" if language == "zh" else "body-en.tex"
    write(PROJECT / body_name, "% Generated index; edit source Markdown.\n\n" +
          "\n".join(rf"\input{{{relative}}}" for _, relative in units) + "\n")
    write(PROJECT / f"sections-{language}.json", json.dumps([asdict(s) for s in sections], ensure_ascii=False, indent=2) + "\n")
    write(PROJECT / f"RESOURCE_MANIFEST{'_EN' if language == 'en' else ''}.md",
          f"# LaTeX resources ({language})\n\n- Sections: {len(sections)}\n- Root documents: {len(root_files)}\n\n"
          "| Source | LaTeX image |\n|---|---|\n" +
          "\n".join(f"| `{source}` | `{target}` |" for source, target in sorted(registry.used.items())) + "\n")


def write_templates(engine: ModuleType, language: str, sections: list[Section], template: str) -> None:
    preamble, ending = template.split(r"\title{", 1)
    preamble += ("\n\\usepackage{needspace}\n"
                 "\\usepackage{placeins}\n"
                 "\\makeatletter\n"
                 "\\pretocmd{\\l@chapter}{\\Needspace{3\\baselineskip}}{}{}\n"
                 "\\makeatother\n")
    ending = r"\title{" + ending
    if language == "en":
        preamble = preamble.replace("pdfauthor={叶逸文}", "pdfauthor={Yiwen Ye},\n  pdflang={en-US}")
        preamble = preamble.replace("name = {第,章}", r"name = {Chapter\space,}")
        preamble += "\n" + "\n".join([r"\renewcommand{\contentsname}{Contents}",
                                        r"\renewcommand{\figurename}{Figure}",
                                        r"\renewcommand{\tablename}{Table}",
                                        r"\renewcommand{\appendixname}{Appendix}"]) + "\n"
        ending = ending.replace("（v2.0）", " (v2.0)").replace("叶逸文", "Yiwen Ye")
        ending = ending.replace("2026年9月2日", "September 2, 2026")
        ending = ending.replace(r"\pdfbookmark[0]{目录}", r"\pdfbookmark[0]{Contents}")
        ending = ending.replace("{body.tex}", "{body-en.tex}")
    else:
        preamble = preamble.replace("pdfauthor={叶逸文}", "pdfauthor={叶逸文},\n  pdflang={zh-CN}")
    write(PROJECT / f"preamble-{language}.tex", preamble)
    main_name = "main.tex" if language == "zh" else "main-en.tex"
    write(PROJECT / main_name, rf"\input{{preamble-{language}.tex}}" + "\n" + ending)
    author = "叶逸文" if language == "zh" else "Yiwen Ye"
    for section in sections:
        esc = engine.latex_escape
        number = f"{int(section.id[:2])}.{int(section.id[3:])}"
        standalone = "\n".join([
            rf"\input{{preamble-{language}.tex}}",
            rf"\hypersetup{{pdftitle={{{esc(number + ' ' + section.title)}}},pdfsubject={{A Comprehensive AI Learning Note v2.0}}}}",
            r"\begin{document}", r"\mainmatter",
            rf"\setcounter{{chapter}}{{{int(section.id[:2])}}}",
            rf"\chaptermark{{{esc(section.chapter_title)}}}",
            r"\begin{center}", rf"{{\small {esc(section.part_title)}\par}}",
            rf"{{\large\bfseries {esc(section.chapter_title)}\par}}",
            rf"{{\small {author} \quad v2.0 \quad {language.upper()}\par}}", r"\end{center}",
            rf"\input{{{section.tex}}}", r"\end{document}", ""])
        write(PROJECT / section.standalone, standalone)


def main(engine: ModuleType) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", default="zh", help="zh, en, or zh,en")
    parser.add_argument("--allow-partial", action="store_true", help="Developer preview; never a complete release")
    args = parser.parse_args()
    languages = args.languages.split(",")
    if not languages or len(set(languages)) != len(languages) or any(x not in {"zh", "en"} for x in languages):
        parser.error("--languages must be zh, en, or zh,en")
    zh = discover(engine, "zh")
    if len(zh) != EXPECTED_SECTIONS:
        raise ValueError(f"Chinese source must have {EXPECTED_SECTIONS} sections, found {len(zh)}")
    selections = {lang: zh if lang == "zh" else discover(engine, lang) for lang in languages}
    if "en" in selections and not args.allow_partial and {s.id for s in selections["en"]} != {s.id for s in zh}:
        raise ValueError("English sections are incomplete or do not map to the Chinese source")
    # Keep the legacy template generator as the single source of proven styles.
    original_main_path = engine.MAIN_TEX
    template_path = PROJECT / "build" / "generated" / "base-template.tex"
    try:
        engine.MAIN_TEX = template_path
        engine.write_main_tex()
    finally:
        engine.MAIN_TEX = original_main_path
    template = engine.read_text(template_path)
    registry = StableImageRegistry()
    for language, sections in selections.items():
        registry.used.clear()
        generate_language(engine, language, sections, registry, args.allow_partial)
        write_templates(engine, language, sections, template)
        print(f"Generated {language}: {len(sections)} shared sections and standalone entries")
    registry.save()
    return 0
