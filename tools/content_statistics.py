"""Reproducible, non-hashing content counts for both editions.

Chinese units count each CJK character and each English/numeric word (including
internal hyphens and underscores) after
removing front matter, references, code, image syntax and link destinations.
English words exclude math as well, and count letter words with internal
apostrophes/hyphens. Neither measure includes root navigation documents.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import zipfile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def prose(text: str, english: bool = False) -> str:
    text = re.sub(r"\A\ufeff?---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.S)
    text = re.split(r"^## (?:参考文献|References)\s*$", text, maxsplit=1, flags=re.M)[0]
    text = re.sub(r"(?m)^\s*(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\s*\1\s*$", "", text)
    text = re.sub(r"!\[[^]]*\]\([^\n]*?\)", "", text)
    if english:
        # GitHub's $`...`$ math must disappear before ordinary code spans.
        # Otherwise two inline formulas turn into $$ ... $$ and consume prose.
        text = re.sub(r"(?<!\\)\$`[^\n]*?`\$", "", text)
    text = re.sub(r"(`+).*?\1", "", text, flags=re.S)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    if english:
        text = re.sub(r"\$\$[\s\S]*?\$\$|(?<!\\)\$[^\n]*?(?<!\\)\$", "", text)
        text = re.sub(r"https?://\S+", "", text)
    return text


def count_text(text: str, language: str) -> int:
    body = prose(text, language == "en")
    if language == "zh":
        return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", body))
    return len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", body))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-root", type=Path, help="Optional local folder containing original Word sources")
    parser.add_argument("--output", type=Path, help="Optional local JSON report; never contains file hashes")
    args = parser.parse_args()
    report = {"methods": {"zh": "CJK characters plus English/numeric words with internal hyphens/underscores; exclude metadata, references, code, images, link targets",
                          "en": "Alphabetic words with internal apostrophes/hyphens; also exclude inline/display math and bare URLs"}}
    sources = set()
    for lang, directory in (("zh", "docs"), ("en", "docs-en")):
        paths = sorted((ROOT / directory).rglob("*.md"))
        parts = {}
        placeholders = 0
        for path in paths:
            text = path.read_text(encoding="utf-8-sig")
            part = path.relative_to(ROOT / directory).parts[0]
            parts[part] = parts.get(part, 0) + count_text(text, lang)
            placeholders += text.count("图片内容待重建")
            if lang == "zh":
                match = re.search(r'^source_docx: "(.+)"$', text, re.M)
                if match:
                    sources.add(match[1])
        report[lang] = {"chapters": len({p.name[:2] for p in paths}), "sections": len(paths),
                        "body_count": sum(parts.values()), "parts": parts, "image_placeholders": placeholders,
                        "section_tex_files": len(list((ROOT / "latex-project" / "content" / lang / "sections").glob("*.tex")))}
    assets = [p for p in (ROOT / "assets" / "images").rglob("*") if p.is_file()]
    report["public_images"] = {"total": len(assets), "english_variants": sum(p.is_relative_to(ROOT / "assets/images/en") for p in assets)}
    report["public_markdown_files"] = len([p for p in ROOT.rglob("*.md") if not any(
        p.relative_to(ROOT).as_posix().startswith(prefix) for prefix in
        (".git/", "local-only/", "latex-project/build/", "output/", "tmp/"))])
    if args.word_root:
        drawings = media = 0
        missing = []
        for source in sorted(sources):
            path = args.word_root / source
            if not path.is_file():
                missing.append(source)
                continue
            with zipfile.ZipFile(path) as package:
                media += sum(name.startswith("word/media/") and not name.endswith("/") for name in package.namelist())
                document = ET.fromstring(package.read("word/document.xml"))
                drawings += sum(1 for item in document.iter() if item.tag.rsplit("}", 1)[-1] in {"blip", "imagedata"})
        report["original_word"] = {"documents": len(sources), "embedded_media_files": media,
                                   "image_references_in_document": drawings, "missing_sources": missing}
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized)


if __name__ == "__main__":
    main()
