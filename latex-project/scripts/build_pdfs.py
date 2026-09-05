"""Build Chinese/English books and individual sections with isolated XeLaTeX jobs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import quote

from pypdf import PdfReader

from bilingual_project import EXPECTED_SECTIONS, PART_NAMES, PROJECT, ROOT, Section

BOOK_NAMES = {
    "zh": "a-comprehensive-ai-learning-note（位于github同名仓库）.pdf",
    "en": "a-comprehensive-ai-learning-note-en.pdf",
}
SECTION_FOLDERS = {"zh": "chinese-sections", "en": "english-sections"}
BAD_LOG = re.compile(
    r"^! |Undefined control sequence|Missing character:|LaTeX Warning:.*(?:undefined|multiply defined)"
    r"|Label\(s\) may have changed|Rerun to get cross-references right", re.M)
OVERFULL = re.compile(r"Overfull \\[hv]box[^\n]*")


def default_output_root(repo: Path = ROOT) -> Path:
    return repo.parent.parent / "output" / "pdf" if repo.parent.name == "github-export" else repo / "output" / "pdf"


def validate_output_root(path: Path) -> Path:
    target = path.expanduser().resolve()
    forbidden = {Path(target.anchor), Path.home().resolve(), ROOT.resolve(), PROJECT.resolve(), ROOT.parent.resolve()}
    if ROOT.parent.name == "github-export":
        forbidden.add(ROOT.parent.parent.resolve())
    if target in forbidden or target.is_relative_to((ROOT / ".git").resolve()):
        raise ValueError(f"Unsafe output root: {target}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Output root is not a directory: {target}")
    return target


def safe_filename(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", text).strip(" .")
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    if not text or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", text.split(".")[0]):
        text = "section-" + text
    return text[:160].rstrip(" .-")


def section_destination(output: Path, section: Section) -> Path:
    if section.part not in PART_NAMES:
        raise ValueError(f"Invalid part in section catalog: {section.part}")
    chapter = safe_filename("chapter-" + section.chapter_directory)[:64].rstrip(" .-")
    title = safe_filename(section.title if section.language == "zh" else section.title.lower())
    directory = output / SECTION_FOLDERS[section.language] / ("part-" + section.part) / chapter
    # Leave room for Windows tools without long-path support. Full titles remain in the index and metadata.
    budget = 240 - len(str(directory.resolve())) - len(f"/{section.id}--{section.language.upper()}.pdf")
    if budget < 16:
        raise ValueError("Output root is too long for portable section PDF paths; choose a shorter --output-root")
    title = title[:budget].rstrip(" .-")
    return directory / f"{section.id}-{title}-{section.language.upper()}.pdf"


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".ainote-copy-", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_copies(source: Path, destinations: list[Path], verify=None) -> None:
    """Publish one verified file to several destinations as a rollback-safe unit."""
    resolved = [destination.resolve() for destination in destinations]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Duplicate publication destination")
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    replaced: list[Path] = []
    retained_backups: set[Path] = set()
    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".ainote-stage-", dir=destination.parent,
                                             delete=False) as handle:
                temporary = Path(handle.name)
            staged[destination] = temporary
            shutil.copy2(source, temporary)
            if verify is not None:
                verify(temporary)
        for destination in destinations:
            backup = None
            if destination.exists():
                with tempfile.NamedTemporaryFile(prefix=".ainote-backup-", dir=destination.parent,
                                                 delete=False) as handle:
                    backup = Path(handle.name)
                backups[destination] = backup
                shutil.copy2(destination, backup)
            else:
                backups[destination] = None
            os.replace(staged[destination], destination)
            replaced.append(destination)
            if verify is not None:
                verify(destination)
    except Exception as publish_error:
        rollback_errors = []
        for destination in reversed(replaced):
            try:
                backup = backups[destination]
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except Exception as rollback_error:
                backup = backups[destination]
                if backup is not None and backup.exists():
                    retained_backups.add(backup)
                    recovery = f"; previous file retained at {backup}"
                else:
                    recovery = "; no previous file was available"
                rollback_errors.append(f"{destination}: {rollback_error}{recovery}")
        if rollback_errors:
            raise RuntimeError("Publication failed and rollback was incomplete: "
                               + "; ".join(rollback_errors)) from publish_error
        raise
    finally:
        temporary_files = [*staged.values(), *(path for path in backups.values() if path is not None)]
        for temporary in temporary_files:
            if temporary not in retained_backups and temporary.exists():
                temporary.unlink()


def atomic_text(destination: Path, content: str) -> None:
    """Keep the previous complete report if writing or replacement fails."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=".ainote-report-", dir=destination.parent,
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def output_lock(reports: Path):
    """OS-released lock prevents two commands from overwriting the shared report."""
    lock = (reports / ".build.lock").open("a+b")
    try:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another build is using this output root; wait for it to finish") from exc
        yield
    finally:
        lock.close()


def pdf_metadata(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty PDF: {path}")
    result = subprocess.run(["pdfinfo", "-enc", "UTF-8", str(path)], capture_output=True,
                            encoding="utf-8", errors="replace", check=True)
    fields = dict(re.findall(r"^([^:\n]+):\s*(.*)$", result.stdout, re.M))
    pages = int(fields.get("Pages", "0"))
    if pages < 1:
        raise ValueError(f"PDF has no pages: {path}")
    with path.open("rb") as stream:
        reader = PdfReader(stream)
        language = str(reader.trailer["/Root"].get("/Lang", ""))
        if len(reader.pages) != pages:
            raise ValueError(f"PDF parsers disagree on page count: {path}")
    return {"pages": pages, "bytes": path.stat().st_size, "title": fields.get("Title", ""),
            "author": fields.get("Author", ""), "page_size": fields.get("Page size", ""),
            "pdf_language": language}


def input_snapshot(tex: Path) -> dict[str, list[int]]:
    """Record local size/mtime, not hashes, for transitive TeX/image inputs."""
    pending, seen = [tex.resolve()], {}
    while pending:
        path = pending.pop()
        relative = path.relative_to(PROJECT.resolve()).as_posix()
        if relative in seen:
            continue
        stat = path.stat()
        seen[relative] = [stat.st_size, stat.st_mtime_ns]
        if path.suffix.lower() != ".tex":
            continue
        content = path.read_text(encoding="utf-8")
        links = re.findall(r"\\input\{([^{}]+)\}|\\includegraphics(?:\[[^]]*\])?\{([^{}]+)\}", content)
        for pair in links:
            link = next(value for value in pair if value)
            if "#" in link:
                continue  # A macro parameter is not a filesystem dependency.
            pending.append((PROJECT / link).resolve())
    return seen


def refresh_state(state: dict, catalogs: dict[str, list[Section]], output: Path) -> None:
    """Do not present an old, removed, relocated or stale PDF as current success."""
    mapped = {lang: {section.id: section for section in sections}
              for lang, sections in catalogs.items()}
    valid = {f"{lang}:{s.id}" for lang, sections in catalogs.items() for s in sections}
    valid.update(f"{lang}:book" for lang in catalogs)
    for key in list(state):
        item = state[key]
        if key not in valid:
            del state[key]  # Only the report entry; never remove a user file.
            continue
        if item.get("status") != "success":
            continue
        try:
            language, item_key = key.split(":", 1)
            if item_key == "book":
                expected_path = output / "complete-books" / BOOK_NAMES[language]
                expected_copy = PROJECT / "build" / BOOK_NAMES[language]
                if Path(item.get("build_copy", "")).resolve() != expected_copy.resolve():
                    raise ValueError("Recorded LaTeX book-copy path is obsolete")
            else:
                expected_path = section_destination(output, mapped[language][item_key])
            if Path(item.get("path", "")).resolve() != expected_path.resolve():
                raise ValueError("Recorded PDF path is obsolete")
            for name in ("path", "build_copy"):
                if name in item:
                    path = Path(item[name])
                    if not path.is_file() or path.stat().st_size != item["bytes"]:
                        raise ValueError(f"Recorded PDF is missing or changed: {path}")
            current = input_snapshot(PROJECT / item["tex"])
            if item.get("inputs") != current:
                raise ValueError("Build inputs changed or are unrecorded; rebuild this PDF")
        except (OSError, ValueError, KeyError) as exc:
            item["status"], item["error"] = "stale", str(exc)


def compile_pdf(language: str, key: str, tex: Path, destination: Path, reports: Path,
                build_copy: Path | None = None) -> dict:
    job = f"{language}-{key}"
    work_root = (PROJECT / "build" / "work").resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"ainote-{job}-", dir=work_root)).resolve()
    if work.parent != work_root or not work.name.startswith("ainote-"):
        raise ValueError("Invalid temporary build directory")
    log_dir = reports / "logs" / job
    log_dir.mkdir(parents=True, exist_ok=True)
    result = {"language": language, "id": key, "status": "failed", "path": str(destination),
              "tex": str(tex.relative_to(PROJECT)), "log": str(log_dir), "warnings": [],
              "built_at": datetime.now(timezone.utc).isoformat()}
    try:
        if not tex.is_file():
            raise FileNotFoundError(f"Generate LaTeX first; missing {tex}")
        inputs = input_snapshot(tex)
        for pass_number in range(1, 4):
            with (log_dir / f"pass-{pass_number}.txt").open("w", encoding="utf-8") as stream:
                completed = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                                            "-file-line-error", f"-jobname={job}",
                                            f"-output-directory={work.as_posix()}", tex.relative_to(PROJECT).as_posix()],
                                           cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"XeLaTeX pass {pass_number} failed ({completed.returncode})")
        log = (work / f"{job}.log").read_text(encoding="utf-8", errors="replace")
        shutil.copy2(work / f"{job}.log", log_dir / "xelatex.log")
        if BAD_LOG.search(log):
            raise RuntimeError("Unresolved error, missing glyph or reference in final XeLaTeX log")
        result["warnings"] = OVERFULL.findall(log)
        if result["warnings"]:
            raise RuntimeError("Overfull content requires layout review before publication")
        built = work / f"{job}.pdf"
        metadata = pdf_metadata(built)
        expected_author = "叶逸文" if language == "zh" else "Yiwen Ye"
        if metadata["author"] != expected_author:
            raise ValueError(f"Unexpected PDF author: {metadata['author']}")
        expected_language = "zh-CN" if language == "zh" else "en-US"
        if metadata["pdf_language"] != expected_language:
            raise ValueError(f"Unexpected PDF language: {metadata['pdf_language']}")
        if key != "book":
            expected_number = ".".join(str(int(part)) for part in key.split("-"))
            if not metadata["title"].startswith(expected_number + " "):
                raise ValueError(f"PDF title does not identify section {key}")
        extracted = subprocess.run(["pdftotext", "-enc", "UTF-8", str(built), "-"],
                                   capture_output=True, encoding="utf-8", errors="strict", check=True).stdout
        if not extracted.strip() or "\ufffd" in extracted:
            raise ValueError("PDF text extraction is empty or has replacement characters")
        compact = re.sub(r"\s+", "", extracted)
        if key != "book" and re.sub(r"\s+", "", metadata["title"]) not in compact:
            raise ValueError("Section heading cannot be found in extracted PDF text")
        if inputs != input_snapshot(tex):
            raise ValueError("Build inputs changed during compilation; retry after generation completes")
        def verify_copy(path: Path) -> None:
            copied = pdf_metadata(path)
            if (copied["bytes"], copied["pages"]) != (metadata["bytes"], metadata["pages"]):
                raise ValueError(f"Published PDF verification failed: {path}")

        destinations = [destination] if build_copy is None else [build_copy, destination]
        atomic_copies(built, destinations, verify_copy)
        if build_copy is not None:
            result["build_copy"] = str(build_copy)
        result.update(metadata)
        result["inputs"] = inputs
        result["status"] = "success"
    except Exception as exc:
        result["error"] = str(exc)
        if (work / f"{job}.log").exists():
            shutil.copy2(work / f"{job}.log", log_dir / "xelatex.log")
    finally:
        # Only remove the exact temporary directory created by this invocation.
        if work.parent == work_root and work.name.startswith(f"ainote-{job}-"):
            shutil.rmtree(work)
    return result


def markdown_link(output: Path, label: str, target: str) -> str:
    path = Path(target).resolve()
    relative = os.path.relpath(path, output).replace("\\", "/")
    return f"[{label.replace('|', '/')}]({quote(relative, safe='/.-_')})"


def write_reports(output: Path, state: dict, catalogs: dict[str, list[Section]], problems: list[str]) -> None:
    reports = output / "build-reports"
    reports.mkdir(parents=True, exist_ok=True)
    state_path = reports / "build-state.json"
    atomic_text(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    lines = ["# PDF Index / PDF 索引", "", "Generated locally. Source Markdown and LaTeX remain in the repository.", "",
             f"Last recorded build: {max((v.get('built_at', '') for v in state.values()), default='not built')}", "",
             "## Complete books / 完整讲义", "", "| Language | PDF | Pages | Status |", "|---|---|---:|---|"]
    for lang in ("zh", "en"):
        item = state.get(f"{lang}:book", {})
        link = markdown_link(output, BOOK_NAMES[lang], item["path"]) if item.get("status") == "success" else "Not built"
        lines.append(f"| {lang} | {link} | {item.get('pages', '')} | {item.get('status', 'not built')} |")
    mapped = {lang: {s.id: s for s in sections} for lang, sections in catalogs.items()}
    ids = sorted(set().union(*(mapping.keys() for mapping in mapped.values())))
    lines += ["", "## Sections / 分节讲义", "", "| ID | Part / Chapter | 中文 | English | ZH PDF / pages | EN PDF / pages |", "|---|---|---|---|---|---|"]
    for sid in ids:
        titles, links = [], []
        for lang in ("zh", "en"):
            section = mapped.get(lang, {}).get(sid)
            titles.append(section.title.replace("|", "/") if section else "")
            item = state.get(f"{lang}:{sid}", {})
            links.append(markdown_link(output, f"{lang.upper()} / {item.get('pages', '')}", item["path"])
                         if item.get("status") == "success" else item.get("status", "not built"))
        section = mapped.get("zh", {}).get(sid) or mapped["en"][sid]
        group = f"{section.part_title} / {int(sid[:2])} {section.chapter_title}".replace("|", "/")
        lines.append(f"| {sid} | {group} | {titles[0]} | {titles[1]} | {links[0]} | {links[1]} |")
    lines += ["", "See [build report](build-reports/BUILD-REPORT.md) for errors and logs.", ""]
    atomic_text(output / "PDF-INDEX.md", "\n".join(lines))
    report = ["# PDF Build Report / PDF 构建报告", "", f"Updated: {datetime.now(timezone.utc).isoformat()}", "",
              "Automated compilation/text checks only. Visual QA and publication are separate acceptance gates.", ""]
    for lang in ("zh", "en"):
        successful = sum(v.get("status") == "success" for k, v in state.items() if k.startswith(lang + ":") and not k.endswith(":book"))
        report.append(f"- {lang} sections: {successful}/{EXPECTED_SECTIONS}")
        book = state.get(f"{lang}:book", {})
        report.append(f"- {lang} book: {book.get('status', 'not built')}; pages: {book.get('pages', '')}; size: {book.get('page_size', '')}")
        if book:
            report.append(f"  - Metadata: title `{book.get('title', '')}`; author `{book.get('author', '')}`; language `{book.get('pdf_language', '')}`")
            report.append(f"  - Output: `{book.get('path', '')}`")
            report.append(f"  - LaTeX build copy: `{book.get('build_copy', '')}`")
    for directory in ("complete-books", "chinese-sections", "english-sections"):
        report.append(f"- Actual PDF files in `{directory}/`: {len(list((output / directory).rglob('*.pdf')))}")
    index_exists = (output / "PDF-INDEX.md").is_file()
    broken = [v["path"] for v in state.values() if v.get("status") == "success" and not Path(v["path"]).is_file()]
    report.append(f"- PDF-INDEX.md: {'present' if index_exists else 'missing'}; broken PDF links: {len(broken)}")
    report += ["- Markdown/LaTeX structural and semantic review: separate validation report required.",
               "- PDF visual review: separate review record required.",
               "- GitHub/Gitee synchronization: not performed by this local build command."]
    errors = [f"{key}: {value.get('error', 'unknown error')} (logs: {value.get('log', '')})"
              for key, value in sorted(state.items()) if value.get("status") != "success"] + problems + broken
    report += ["", "## Failures", ""] + (["- " + error for error in errors] if errors else ["None in the recorded jobs."])
    report += ["", "## Logs", "", "Per-job logs are under `logs/<language>-<section-or-book>/`.", ""]
    atomic_text(reports / "BUILD-REPORT.md", "\n".join(report))
    atomic_text(reports / "failed-sections.txt", "\n".join(errors) + "\n" if errors else "none\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["book", "section", "sections", "all"], default="all")
    parser.add_argument("--language", choices=["zh", "en", "all"], default="all")
    parser.add_argument("--section", help="Stable section ID, e.g. 01-03")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.scope == "section" and (not args.section or not re.fullmatch(r"\d{2}-\d{2}", args.section)):
        parser.error("--scope section requires --section NN-NN")
    if args.scope != "section" and args.section:
        parser.error("--section is only valid with --scope section")
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be between 1 and 8")
    for command in ("xelatex", "pdfinfo", "pdftotext"):
        if not shutil.which(command):
            parser.error(f"Missing dependency: {command}")
    output = validate_output_root(args.output_root or default_output_root())
    output.mkdir(parents=True, exist_ok=True)
    reports = output / "build-reports"
    reports.mkdir(parents=True, exist_ok=True)
    print(f"PDF output: {output}", flush=True)
    catalogs = {}
    for lang in ("zh", "en"):
        manifest = PROJECT / f"sections-{lang}.json"
        if manifest.exists():
            catalogs[lang] = [Section(**item) for item in json.loads(manifest.read_text(encoding="utf-8"))]
    languages = ["zh", "en"] if args.language == "all" else [args.language]
    tasks = []
    for lang in languages:
        sections = catalogs.get(lang, [])
        if not sections:
            parser.error(f"Generate {lang} LaTeX first")
        ids = [s.id for s in sections]
        if len(ids) != len(set(ids)):
            parser.error(f"Duplicate section IDs in {lang} catalog")
        for section in sections:
            if section.language != lang or not re.fullmatch(r"\d{2}-\d{2}", section.id):
                parser.error(f"Invalid catalog entry: {lang} {section.id}")
            for relative in (section.markdown, "latex-project/" + section.tex, "latex-project/" + section.standalone):
                source = (ROOT / relative).resolve()
                if not source.is_relative_to(ROOT.resolve()) or not source.is_file():
                    parser.error(f"Missing or unsafe source path: {relative}")
        if args.scope != "section" and len(sections) != EXPECTED_SECTIONS:
            parser.error(f"{lang} requires {EXPECTED_SECTIONS} sections, found {len(sections)}")
        if args.scope in {"book", "all"}:
            tasks.append((lang, "book", PROJECT / ("main.tex" if lang == "zh" else "main-en.tex"),
                          output / "complete-books" / BOOK_NAMES[lang], reports, PROJECT / "build" / BOOK_NAMES[lang]))
        if args.scope != "book":
            selected = [s for s in sections if args.scope != "section" or s.id == args.section]
            if not selected:
                parser.error(f"Section not found: {lang} {args.section}")
            for section in selected:
                tasks.append((lang, section.id, PROJECT / section.standalone, section_destination(output, section), reports, None))
    destinations = [str(task[3]).casefold() for task in tasks]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Conflicting PDF output filenames")
    with output_lock(reports):
        return build_jobs(output, catalogs, tasks, languages, args.scope, args.jobs)


def build_jobs(output: Path, catalogs: dict, tasks: list, languages: list[str], scope: str, jobs: int) -> int:
    reports = output / "build-reports"
    state_file = reports / "build-state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    refresh_state(state, catalogs, output)
    failed = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(compile_pdf, *task) for task in tasks]
        for future in as_completed(futures):
            item = future.result()
            state[f"{item['language']}:{item['id']}"] = item
            failed += item["status"] != "success"
            print(f"{item['language']} {item['id']}: {item['status']} {item.get('error', '')}", flush=True)
            write_reports(output, state, catalogs, [])
    problems = []
    refresh_state(state, catalogs, output)
    for task in tasks:
        item = state.get(f"{task[0]}:{task[1]}", {})
        if item.get("status") != "success":
            problems.append(f"{task[0]} {task[1]} did not produce a current verified PDF")
    if scope in {"sections", "all"}:
        for lang in languages:
            actual = set((output / SECTION_FOLDERS[lang]).rglob("*.pdf"))
            expected = {section_destination(output, s) for s in catalogs[lang]}
            if actual != expected:
                problems.append(f"{lang} output inventory mismatch: missing {len(expected-actual)}, extra {len(actual-expected)}")
    write_reports(output, state, catalogs, problems)
    print(f"Finished: {len(tasks)-failed}/{len(tasks)} jobs; output: {output}", flush=True)
    return 1 if failed or problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
