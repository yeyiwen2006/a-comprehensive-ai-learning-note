#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上传前验证脚本。

目标不是证明内容完全正确，而是阻止明显错误：
1. Word 原稿、OCR 临时文件和未经筛选的本地图片素材不能进入上传仓库。
2. Markdown 文件不能为空，不能出现明显乱码。
3. 目录中的链接应指向存在的文件。
4. Markdown 数学公式应使用 GitHub 可渲染分隔符。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote


PUBLIC_IMAGE_ROOT = Path("assets") / "images"
LATEX_IMAGE_ROOT = Path("latex-project") / "images"
ALLOWED_IMAGE_ROOTS = (PUBLIC_IMAGE_ROOT, LATEX_IMAGE_ROOT)

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

BANNED_SUFFIXES = {
    ".docx",
    ".doc",
    ".emf",
    ".wmf",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate generated Markdown repository.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    return parser.parse_args()


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"[FAIL] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fenced_code_lines(lines: list[str]) -> list[bool]:
    """按 CommonMark 围栏规则标记代码围栏及其内容所在的行。"""

    masked = [False] * len(lines)
    fence: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        if fence is not None:
            masked[index] = True
            closing = re.match(r"^ {0,3}(`+|~+)[ \t]*$", line)
            if closing:
                marker = closing.group(1)
                if marker[0] == fence[0] and len(marker) >= fence[1]:
                    fence = None
            continue

        opening = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if not opening:
            continue
        marker, info = opening.groups()
        if marker[0] == "`" and "`" in info:
            continue
        masked[index] = True
        fence = (marker[0], len(marker))

    return masked


def is_escaped(text: str, index: int) -> bool:
    """判断指定字符前是否有奇数个连续反斜杠。"""

    slashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slashes += 1
        cursor -= 1
    return slashes % 2 == 1


def mask_matching_code_spans(segment: str) -> str:
    """屏蔽一个 Markdown 块内由等长反引号包围的完整代码 span。"""

    chars = list(segment)
    runs = list(re.finditer(r"`+", segment))
    run_index = 0
    while run_index < len(runs):
        opening = runs[run_index]
        if is_escaped(segment, opening.start()):
            run_index += 1
            continue

        closing_index = run_index + 1
        while closing_index < len(runs):
            closing = runs[closing_index]
            if len(closing.group(0)) == len(opening.group(0)):
                for position in range(opening.start(), closing.end()):
                    if chars[position] != "\n":
                        chars[position] = " "
                run_index = closing_index + 1
                break
            closing_index += 1
        else:
            run_index += 1

    return "".join(chars)


def is_plain_paragraph_line(line: str) -> bool:
    """判断一行能否与相邻普通文本行共同组成跨行 code span。"""

    if not line.strip() or re.match(r"^(?: {4}|\t)", line):
        return False
    if re.match(r"^ {0,3}(?:#{1,6}(?:[ \t]+|$)|>|(?:[-+*]|\d{1,9}[.)])[ \t]+)", line):
        return False
    if re.match(r"^ {0,3}(?:([-*_])[ \t]*){3,}$", line):
        return False
    if re.match(r"^ {0,3}(?:=+|-+)[ \t]*$", line):
        return False
    if re.match(r"^ {0,3}(?:\[[^]]+\]:|\|)", line):
        return False
    if re.match(r"^ {0,3}(?:<!--|</?[A-Za-z][^>]*>)", line):
        return False
    return True


def mask_prefixed_code_span_block(
    result: list[str], indexes: list[int], prefix_lengths: list[int]
) -> None:
    """剥离同一容器块的 Markdown 前缀后，屏蔽其中的跨行 code span。"""

    segment = "\n".join(
        result[index][prefix_length:]
        for index, prefix_length in zip(indexes, prefix_lengths, strict=True)
    )
    masked = mask_matching_code_spans(segment).split("\n")
    for index, prefix_length, content in zip(indexes, prefix_lengths, masked, strict=True):
        result[index] = result[index][:prefix_length] + content


def mask_container_code_spans(
    lines: list[str], fenced: list[bool], result: list[str]
) -> list[bool]:
    """处理 blockquote 和单个列表项内部合法的跨行 code span。"""

    container = [False] * len(lines)
    index = 0
    while index < len(lines):
        if fenced[index] or not lines[index].strip():
            index += 1
            continue

        quote = re.match(r"^ {0,3}>[ \t]?(.*)$", lines[index])
        if quote:
            indexes: list[int] = []
            prefixes: list[int] = []
            end = index
            while end < len(lines) and not fenced[end] and lines[end].strip():
                current = re.match(r"^ {0,3}>[ \t]?(.*)$", lines[end])
                if current:
                    if not is_plain_paragraph_line(current.group(1)):
                        break
                    prefix_length = current.start(1)
                elif is_plain_paragraph_line(lines[end]):
                    prefix_length = 0
                else:
                    break
                indexes.append(end)
                prefixes.append(prefix_length)
                container[end] = True
                end += 1
            if not indexes:
                index += 1
                continue
            mask_prefixed_code_span_block(result, indexes, prefixes)
            index = end
            continue

        item = re.match(r"^ {0,3}(?:[-+*]|\d{1,9}[.)])[ \t]+(.*)$", lines[index])
        if item:
            content_indent = item.start(1)
            indexes = [index]
            prefixes = [content_indent]
            container[index] = True
            end = index + 1
            while end < len(lines) and not fenced[end] and lines[end].strip():
                indentation = len(lines[end]) - len(lines[end].lstrip(" "))
                if indentation >= content_indent:
                    if not is_plain_paragraph_line(lines[end][content_indent:]):
                        break
                    prefix_length = content_indent
                elif is_plain_paragraph_line(lines[end]):
                    prefix_length = 0
                else:
                    break
                indexes.append(end)
                prefixes.append(prefix_length)
                container[end] = True
                end += 1
            mask_prefixed_code_span_block(result, indexes, prefixes)
            index = end
            continue

        index += 1

    return container


def mask_inline_code_spans(lines: list[str], fenced: list[bool]) -> list[str]:
    """屏蔽完整的 CommonMark 行内代码，并避免跨 Markdown 块误配。"""

    result = [
        line if fenced[index] else mask_matching_code_spans(line)
        for index, line in enumerate(lines)
    ]
    container = mask_container_code_spans(lines, fenced, result)

    index = 0
    while index < len(lines):
        if fenced[index] or container[index] or not is_plain_paragraph_line(lines[index]):
            index += 1
            continue

        end = index + 1
        while (
            end < len(lines)
            and not fenced[end]
            and not container[end]
            and is_plain_paragraph_line(lines[end])
        ):
            end += 1

        segment = "\n".join(result[index:end])
        result[index:end] = mask_matching_code_spans(segment).split("\n")
        index = end

    return result


def validate_github_math_blocks(path: Path, repo_root: Path, text: str, failures: list[str]) -> None:
    """检查 $$ 块级公式是否按 GitHub 能稳定渲染的方式独立成段。"""

    lines = text.splitlines()
    fenced = fenced_code_lines(lines)
    math_lines = mask_inline_code_spans(lines, fenced)
    in_math_block = False
    for index, line in enumerate(lines):
        if fenced[index]:
            continue

        math_line = math_lines[index]
        if "$$" in math_line and math_line.strip() != "$$":
            fail(
                f"块级数学公式标记 $$ 必须独占一行，避免 GitHub 解析失败: {path.relative_to(repo_root)}:{index + 1}",
                failures,
            )
            continue

        if math_line.strip() != "$$":
            continue

        line_no = index + 1
        previous_blank = index == 0 or not lines[index - 1].strip()
        next_blank = index == len(lines) - 1 or not lines[index + 1].strip()

        if not in_math_block:
            if not previous_blank:
                fail(
                    f"块级数学公式开始标记 $$ 前需要空行，避免 GitHub 解析失败: {path.relative_to(repo_root)}:{line_no}",
                    failures,
                )
            in_math_block = True
        else:
            if not next_blank:
                fail(
                    f"块级数学公式结束标记 $$ 后需要空行，避免 GitHub 解析失败: {path.relative_to(repo_root)}:{line_no}",
                    failures,
                )
            in_math_block = False

    if in_math_block:
        fail(f"块级数学公式 $$ 标记未闭合: {path.relative_to(repo_root)}", failures)


def validate_no_banned_files(repo_root: Path, failures: list[str]) -> None:
    local_only_warned = False
    for path in repo_root.rglob("*"):
        relative = path.relative_to(repo_root)
        if ".git" in relative.parts:
            continue
        if "local-only" in relative.parts:
            if not local_only_warned:
                warn("检测到本地-only目录，必须保持 gitignore 不上传: local-only")
                local_only_warned = True
            continue
        if not path.is_file():
            if path.is_dir() and path.name in {"_ocr_tmp", "_tmp"}:
                fail(f"禁止上传的临时或本地目录: {path.relative_to(repo_root)}", failures)
            continue

        suffix = path.suffix.lower()
        relative_posix = str(relative).replace("\\", "/")
        allowed_image_roots = tuple(str(root).replace("\\", "/") + "/" for root in ALLOWED_IMAGE_ROOTS)
        if suffix in IMAGE_SUFFIXES and not relative_posix.startswith(allowed_image_roots):
            fail(f"图片只能放在公开资源目录 assets/images 下: {path.relative_to(repo_root)}", failures)
        if suffix in BANNED_SUFFIXES:
            fail(f"禁止上传的文件类型: {path.relative_to(repo_root)}", failures)


def validate_markdown_files(repo_root: Path, failures: list[str]) -> None:
    markdown_files = list(repo_root.rglob("*.md"))
    if not markdown_files:
        fail("仓库中没有 Markdown 文件。", failures)
        return

    for path in markdown_files:
        relative_posix = str(path.relative_to(repo_root)).replace("\\", "/")
        if relative_posix.startswith("local-only/"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            fail(f"Markdown 空文件: {path.relative_to(repo_root)}", failures)
        if "\ufffd" in text or "????" in text:
            fail(f"疑似编码乱码: {path.relative_to(repo_root)}", failures)
        if any(marker in text for marker in ("\\[", "\\]", "\\(", "\\)")):
            fail(
                f"检测到 GitHub 不兼容的数学公式分隔符，请使用 $...$ 或 $$...$$: {path.relative_to(repo_root)}",
                failures,
            )
        validate_github_math_blocks(path, repo_root, text, failures)


def validate_required_files(repo_root: Path, failures: list[str]) -> None:
    required = [
        "README.md",
        "LICENSE",
        ".gitignore",
        "CONTRIBUTING.md",
        "DISCLAIMER.md",
        "CITATION.cff",
        "CHANGELOG.md",
        "目录.md",
        "初学者学习路径.md",
    ]
    for relative in required:
        if not (repo_root / relative).exists():
            fail(f"缺少必需文件: {relative}", failures)


def validate_directory_links(repo_root: Path, failures: list[str]) -> None:
    directory = repo_root / "目录.md"
    if not directory.exists():
        return
    text = directory.read_text(encoding="utf-8", errors="replace")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    for link in links:
        if link.startswith("http://") or link.startswith("https://"):
            continue
        target = repo_root / unquote(link)
        if not target.exists():
            fail(f"目录链接指向不存在的文件: {link}", failures)


def validate_license(repo_root: Path, failures: list[str]) -> None:
    license_path = repo_root / "LICENSE"
    if not license_path.exists():
        return
    text = license_path.read_text(encoding="utf-8", errors="replace")
    if "Attribution-NonCommercial-ShareAlike 4.0" not in text:
        fail("LICENSE 中未检测到 CC BY-NC-SA 4.0 关键字。", failures)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    failures: list[str] = []

    validate_required_files(repo_root, failures)
    validate_no_banned_files(repo_root, failures)
    validate_markdown_files(repo_root, failures)
    validate_directory_links(repo_root, failures)
    validate_license(repo_root, failures)

    docs_count = len(list((repo_root / "docs").rglob("*.md"))) if (repo_root / "docs").exists() else 0
    if docs_count == 0:
        fail("docs 目录中没有上传版 Markdown。", failures)
    else:
        print(f"[OK] docs Markdown files: {docs_count}")

    if failures:
        print(f"[SUMMARY] validation failed: {len(failures)} issue(s)")
        return 1
    print("[SUMMARY] validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
