"""Rebuild English navigation from the Chinese link order and translated titles.

This script does not translate prose or section content. The two introductions
and the learning-path conclusion below are reviewed translations of the source.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PARTS = {
    1: "Deep Learning",
    2: "Reinforcement Learning",
    3: "Large Language Models",
    4: "LLM Agents",
    5: "Diffusion Models and Multimodal Generation",
    6: "Embodied AI and World Models",
}


def translated_sections() -> dict[str, tuple[str, str, str]]:
    result = {}
    for path in sorted((ROOT / "docs-en").rglob("*.md")):
        section_id = path.name[:5]
        text = path.read_text(encoding="utf-8")
        heading = re.search(r"^# (.+)$", text, re.M)
        chapter = re.search(r'^chapter_title:\s*"([^"]+)"\s*$', text, re.M)
        if not re.fullmatch(r"\d{2}-\d{2}", section_id) or not heading or not chapter or section_id in result:
            raise ValueError(f"Invalid or duplicate section: {path}")
        result[section_id] = (heading[1], path.relative_to(ROOT).as_posix(), chapter[1])
    return result


def translated_link(line: str, sections: dict) -> str:
    match = re.fullmatch(r"(\d+\. |- )\[.+\]\((.+)\)", line)
    if not match:
        raise ValueError(f"Unrecognized source navigation line: {line}")
    source = ROOT / unquote(match[2])
    if not source.is_file():
        raise FileNotFoundError(source)
    title, target, _ = sections[source.name[:5]]
    return f"{match[1]}[{title}]({target})"


def main() -> None:
    sections = translated_sections()
    chapter_titles: dict[str, str] = {}
    for section_id, (_, _, chapter_title) in sections.items():
        chapter_id = section_id[:2]
        if chapter_id in chapter_titles and chapter_titles[chapter_id] != chapter_title:
            raise ValueError(f"Inconsistent English chapter title for Chapter {int(chapter_id)}")
        chapter_titles[chapter_id] = chapter_title
    source_ids = {p.name[:5] for p in (ROOT / "docs").rglob("*.md")}
    if len(source_ids) != 168 or set(sections) != source_ids:
        raise ValueError("Both editions must contain the same 168 sections before navigation is generated")
    source = (ROOT / "初学者学习路径.md").read_text(encoding="utf-8")
    links = [translated_link(line, sections) for line in source.splitlines() if re.match(r"\d+\. \[", line)]
    if len(links) != 24:
        raise ValueError("Review the changed Chinese beginner learning path before regenerating")
    beginner = ["# Beginner Learning Path", "",
                "This file is for beginners visiting the repository for the first time. We recommend reading in the following order:",
                "", *links, "",
                "After completing the material above, you can explore other chapters that interest you. See the [Table of Contents](TABLE_OF_CONTENTS_EN.md) for the full content.", ""]
    catalog = ["# Table of Contents", "", "This table of contents is generated from the current public Markdown documents.", ""]
    catalog_source = (ROOT / "目录.md").read_text(encoding="utf-8")
    linked_ids = []
    for line in catalog_source.splitlines():
        if line.startswith("## "):
            part = re.fullmatch(r"## 第(\d+)部分 .+", line)
            if not part:
                raise ValueError(f"Unrecognized part heading: {line}")
            catalog += [f"## Part {part[1]} {PARTS[int(part[1])]}", ""]
        elif line.startswith("### "):
            chapter = re.fullmatch(r"### 第(\d+)章 .+", line)
            if not chapter:
                raise ValueError(f"Unrecognized chapter heading: {line}")
            chapter_id = f"{int(chapter[1]):02d}"
            if chapter_id not in chapter_titles:
                raise ValueError(f"Missing English title for Chapter {int(chapter[1])}")
            catalog += [f"### Chapter {int(chapter[1])}: {chapter_titles[chapter_id]}", ""]
        elif line.startswith("- ["):
            catalog.append(translated_link(line, sections))
            linked_ids.append(Path(unquote(re.search(r"\]\((.+)\)", line)[1])).name[:5])
        elif not line.strip() and catalog[-1] != "":
            catalog.append("")
    if len(linked_ids) != 168 or set(linked_ids) != source_ids:
        raise ValueError("Chinese catalog does not contain every section exactly once")
    for name, lines in [("BEGINNER_LEARNING_PATH_EN.md", beginner), ("TABLE_OF_CONTENTS_EN.md", catalog)]:
        text = "\n".join(lines).rstrip() + "\n"
        destination = ROOT / name
        if not destination.exists() or destination.read_text(encoding="utf-8") != text:
            destination.write_text(text, encoding="utf-8", newline="\n")
        print(f"Generated {name}")


if __name__ == "__main__":
    main()
