"""Read-only repository assertions and isolated temporary-file unit tests."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote

import bilingual_project as generator
import build_latex_project as legacy
import build_pdfs as builder

sys.path.insert(0, str(generator.ROOT / "tools"))
import content_statistics as statistics
import validate_generated_markdown as validator


class StatisticsTests(unittest.TestCase):
    def test_github_math_does_not_consume_surrounding_prose(self):
        text = "Before $`x`$ middle words $`y`$ after"
        self.assertEqual(statistics.count_text(text, "en"), 4)
        self.assertIn("middle words", statistics.prose(text, True))

    def test_english_excludes_math_code_references_and_metadata(self):
        text = "---\ntitle: metadata words\n---\n# Heading\nBefore $x+y$ after.\n$$\n\\text{math words}\n$$\n`code words`\n## References\nExcluded reference words"
        self.assertEqual(statistics.count_text(text, "en"), 3)

    def test_chinese_count_keeps_existing_convention(self):
        text = "正文 two_words 2026 $x$"
        self.assertEqual(statistics.count_text(text, "zh"), 5)


class ValidatorTests(unittest.TestCase):
    def test_missing_or_unterminated_front_matter_reports_failure(self):
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            root = Path(folder)
            section = root / "docs-en" / "01-01-test.md"
            section.parent.mkdir()
            for text in ("# No front matter\n", "---\nlanguage: en\n# No closing marker\n"):
                section.write_text(text, encoding="utf-8")
                failures = []
                with patch("builtins.print"):
                    validator.validate_bilingual_mapping(root, failures, True)
                self.assertTrue(any("缺少完整元数据前言" in error for error in failures))


class GeneratorTests(unittest.TestCase):
    def test_exact_unit_split(self):
        self.assertEqual(generator.split_units(
            "% AINOTE-UNIT one\nA\n% AINOTE-UNIT two\nB\n",
            [("one", "one.tex"), ("two", "two.tex")]),
            {"one.tex": "A\n", "two.tex": "B\n"})

    def test_missing_duplicate_reordered_and_unassigned_markers(self):
        expected = [("one", "one.tex"), ("two", "two.tex")]
        for value in ("% AINOTE-UNIT one\nA", "% AINOTE-UNIT one\n% AINOTE-UNIT one",
                      "% AINOTE-UNIT two\n% AINOTE-UNIT one",
                      "unassigned\n% AINOTE-UNIT one\n% AINOTE-UNIT two"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generator.split_units(value, expected)

    def test_chinese_catalog(self):
        sections = generator.discover(legacy, "zh")
        self.assertEqual(len(sections), 168)
        self.assertEqual(len({s.id[:2] for s in sections}), 35)
        self.assertEqual(len({s.part for s in sections}), 6)
        for section in sections:
            standalone = (generator.PROJECT / section.standalone).read_text(encoding="utf-8")
            self.assertIn(r"\input{" + section.tex + "}", standalone)
            self.assertNotIn(r"\input{body.tex}", standalone)
            self.assertTrue((generator.PROJECT / section.tex).is_file())

    def test_existing_english_ids_are_subset(self):
        chinese = {s.id for s in generator.discover(legacy, "zh")}
        english = {s.id for s in generator.discover(legacy, "en")}
        self.assertLessEqual(english, chinese)

    def test_english_vector_reflow_preserves_all_values(self):
        vector = "[0.214, -0.419, 0.733, -0.057, 0.024, 0.471, -0.157, -1.0]"
        result = generator.repaired_tex(legacy, "Normalized tensor: " + r"\(" + vector + r"\)")
        self.assertIn("\\[\n" + vector + "\n\\]", result)
        self.assertEqual(result.count(vector), 1)
        self.assertNotRegex(result, r"(?m)[ \t]+$")

    def test_standalone_list_heading_reserves_following_space(self):
        listing = "\\begin{enumerate}\n\\setcounter{enumi}{1}\n\\tightlist\n\\item\n  Input\n\\end{enumerate}\n\nExplanation."
        result = generator.repaired_tex(legacy, listing)
        self.assertTrue(result.startswith("\\Needspace{4\\baselineskip}\n"))
        self.assertIn(listing, result)
        self.assertEqual(generator.repaired_tex(legacy, result), result)
        sentence = listing.replace("  Input\n", "  A complete sentence.\n")
        self.assertNotIn("\\Needspace", generator.repaired_tex(legacy, sentence))

    def test_plain_chinese_prose_headings_reserve_following_space(self):
        for heading in ("（3）现代改进", "（5）加入正则化后", "（3）冻结预训练模型的主体的权重"):
            source = f"Before.\n\n{heading}\n\nExplanation."
            result = generator.repaired_tex(legacy, source)
            self.assertIn(f"\\Needspace{{8\\baselineskip}}\n{heading}", result)
            self.assertEqual(result.count("\\Needspace{8\\baselineskip}"), 1)
            self.assertEqual(generator.repaired_tex(legacy, result), result)

    def test_plain_ascii_step_labels_and_short_code_are_kept_together(self):
        for heading in ("2. Why does this help?", "Step 3: Aggregate statistics",
                        "Step 2：Block-wise segmentation.", "步骤 3：混合前向传播"):
            result = generator.repaired_tex(legacy, f"Before.\n\n{heading}\n\nBody.")
            self.assertIn(f"\\Needspace{{8\\baselineskip}}\n{heading}", result)
        short = "\\begin{verbatim}\nfor x in xs:\n    print(x)\n\\end{verbatim}"
        guarded = generator.protect_short_verbatim(short)
        self.assertEqual(guarded.count("\\Needspace{10\\baselineskip}"), 1)
        self.assertEqual(generator.protect_short_verbatim(guarded), guarded)
        long_block = "\\begin{verbatim}\n" + "\n".join(str(i) for i in range(9)) + "\n\\end{verbatim}"
        self.assertEqual(generator.protect_short_verbatim(long_block), long_block)
        python_block = "  \\begin{verbatim}\nfor _ in range(10000):\n    y = x + 1\n  \\end{verbatim}"
        self.assertNotIn("\\Needspace", generator.protect_layout_boundaries(python_block))

    def test_reference_barrier_and_dqn_image_width(self):
        reference = "\\hypertarget{refs}{%\n\\subsection{References}\\label{refs}}"
        image = "\\includegraphics{images/image-0014.jpeg}"
        result = generator.repaired_tex(legacy, image + "\n\n" + reference)
        self.assertIn("\\includegraphics[width=0.82\\linewidth]{images/image-0014.jpeg}", result)
        self.assertIn("\\FloatBarrier\n\\Needspace{5\\baselineskip}\n" + reference, result)
        self.assertEqual(generator.repaired_tex(legacy, result), result)

    def test_front_heading_guard_precedes_hyperlink_target(self):
        content = "\\hypertarget{date}{%\n\\section{Date}\\label{date}}\n\nBody."
        guarded = generator.protect_frontmatter_headings(content)
        self.assertEqual(guarded, "\\Needspace{5\\baselineskip}\n" + content)
        self.assertEqual(guarded.count("\\Needspace"), 1)
        self.assertEqual(generator.protect_frontmatter_headings(guarded), guarded)

    def test_structural_heading_and_short_lead_reserve_following_space(self):
        heading = "\\hypertarget{topic}{%\n\\subsection{Topic}\\label{topic}}\n\nBody."
        lead = "Before.\n\nDefine the quadratic function:\n\n\\[\nf(x)=x^2\n\\]"
        list_lead = "Before.\n\nThe two methods are:\n\n\\begin{enumerate}\n\\item One\n\\end{enumerate}"
        for source, boundary in ((heading, "\\hypertarget{topic}"),
                                 (lead, "Define the quadratic function:"),
                                 (list_lead, "The two methods are:")):
            guarded = generator.protect_layout_boundaries(source)
            self.assertIn("\\Needspace{5\\baselineskip}\n" + boundary, guarded)
            self.assertEqual(guarded.count("\\Needspace{5\\baselineskip}"), 1)
            self.assertEqual(generator.protect_layout_boundaries(guarded), guarded)

    def test_generated_preamble_protects_chapter_toc_entries(self):
        template_path = generator.PROJECT / "build" / "generated" / "unit-template.tex"
        old_main = legacy.MAIN_TEX
        try:
            legacy.MAIN_TEX = template_path
            legacy.write_main_tex()
        finally:
            legacy.MAIN_TEX = old_main
        template = template_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            with patch.object(generator, "PROJECT", Path(folder)):
                generator.write_templates(legacy, "zh", [], template)
                preamble = (Path(folder) / "preamble-zh.tex").read_text(encoding="utf-8")
        self.assertIn(r"\pretocmd{\l@chapter}{\Needspace{3\baselineskip}}{}{}", preamble)


class BuilderTests(unittest.TestCase):
    def test_default_workspace_and_standalone_output(self):
        base = Path(tempfile.gettempdir()) / "ainote-unit"
        self.assertEqual(builder.default_output_root(base / "github-export" / "repo"), base / "output" / "pdf")
        self.assertEqual(builder.default_output_root(base / "repo"), base / "repo" / "output" / "pdf")

    def test_unsafe_output_roots(self):
        for path in (Path.home(), builder.ROOT, builder.PROJECT, Path(builder.ROOT.anchor), builder.ROOT / ".git" / "test"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                builder.validate_output_root(path)

    def test_windows_filename_rules(self):
        for source in ("CON", "nul.txt", "aux", "LPT1", "a<b>c:d\"e/f\\g|h?i*", " tail. ", "", "a" * 500):
            name = builder.safe_filename(source)
            self.assertTrue(name)
            self.assertLessEqual(len(name), 160)
            self.assertNotRegex(name, r'[<>:"/\\|?*\x00-\x1f]')
            self.assertFalse(name.endswith((" ", ".")))
        self.assertEqual(builder.safe_filename("机器学习的分类"), "机器学习的分类")

    def test_unique_section_destinations(self):
        sections = generator.discover(legacy, "zh")
        paths = [builder.section_destination(Path("output"), s) for s in sections]
        self.assertEqual(len(set(str(p).casefold() for p in paths)), 168)
        for section, path in zip(sections, paths, strict=True):
            self.assertTrue(path.name.startswith(section.id + "-"))
            self.assertTrue(path.name.endswith("-ZH.pdf"))
            self.assertEqual(path.parts[1], "chinese-sections")

    def test_tex_diagnostic_is_not_fatal(self):
        self.assertIsNone(builder.BAD_LOG.search("![] \\OT1/cmr/m/n/10.95 ="))
        for text in ("! Undefined control sequence.", "Missing character: There is no 字",
                     "LaTeX Warning: Reference `x' undefined", "Label(s) may have changed"):
            self.assertIsNotNone(builder.BAD_LOG.search(text))

    def test_relative_index_link(self):
        base = Path(tempfile.gettempdir()) / "ainote-link-test"
        target = base / "chinese-sections" / "01-01-测试-ZH.pdf"
        link = builder.markdown_link(base, "ZH / 1", str(target))
        self.assertEqual(unquote(link.split("](")[1][:-1]), "chinese-sections/01-01-测试-ZH.pdf")
        self.assertNotIn(str(base), link)

    def test_atomic_copy(self):
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            base = Path(folder)
            source, destination = base / "new.pdf", base / "out" / "book.pdf"
            source.write_bytes(b"new-pdf-fixture")
            builder.atomic_copy(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_atomic_copies_roll_back_all_destinations(self):
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            base = Path(folder)
            source = base / "new.pdf"
            first, second = base / "build" / "book.pdf", base / "output" / "book.pdf"
            source.write_bytes(b"new-verified")
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"old-build")
            second.write_bytes(b"old-output")
            real_replace = builder.os.replace

            def fail_second(source_path, destination_path):
                if Path(destination_path) == second:
                    raise PermissionError("locked output")
                return real_replace(source_path, destination_path)

            with patch.object(builder.os, "replace", side_effect=fail_second):
                with self.assertRaises(PermissionError):
                    builder.atomic_copies(source, [first, second])
            self.assertEqual(first.read_bytes(), b"old-build")
            self.assertEqual(second.read_bytes(), b"old-output")
            self.assertEqual(list(first.parent.iterdir()), [first])
            self.assertEqual(list(second.parent.iterdir()), [second])

    def test_atomic_copies_retain_backup_when_rollback_fails(self):
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            base = Path(folder)
            source = base / "new.pdf"
            first, second = base / "build" / "book.pdf", base / "output" / "book.pdf"
            source.write_bytes(b"new-verified")
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"old-build")
            second.write_bytes(b"old-output")
            real_replace = builder.os.replace

            def fail_publish_and_rollback(source_path, destination_path):
                source_path, destination_path = Path(source_path), Path(destination_path)
                if destination_path == second:
                    raise PermissionError("locked output")
                if destination_path == first and source_path.name.startswith(".ainote-backup-"):
                    raise PermissionError("locked rollback")
                return real_replace(source_path, destination_path)

            with patch.object(builder.os, "replace", side_effect=fail_publish_and_rollback):
                with self.assertRaisesRegex(RuntimeError, "previous file retained at"):
                    builder.atomic_copies(source, [first, second])
            self.assertEqual(first.read_bytes(), b"new-verified")
            self.assertEqual(second.read_bytes(), b"old-output")
            retained = list(first.parent.glob(".ainote-backup-*"))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_bytes(), b"old-build")
            self.assertEqual(list(second.parent.iterdir()), [second])

    def test_atomic_copies_clean_partial_stage_and_backup_files(self):
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            base = Path(folder)
            source, destination = base / "new.pdf", base / "out" / "book.pdf"
            source.write_bytes(b"new")
            with patch.object(builder.shutil, "copy2", side_effect=OSError("stage copy failed")):
                with self.assertRaises(OSError):
                    builder.atomic_copies(source, [destination])
            self.assertEqual(list(destination.parent.iterdir()), [])

            destination.write_bytes(b"old")
            real_copy = builder.shutil.copy2

            def fail_backup(source_path, destination_path):
                if Path(destination_path).name.startswith(".ainote-backup-"):
                    raise OSError("backup copy failed")
                return real_copy(source_path, destination_path)

            with patch.object(builder.shutil, "copy2", side_effect=fail_backup):
                with self.assertRaises(OSError):
                    builder.atomic_copies(source, [destination])
            self.assertEqual(destination.read_bytes(), b"old")
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_refresh_state_rejects_obsolete_section_destination(self):
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            output = Path(folder) / "pdf"
            section = builder.Section("01-01", "zh", "docs/example.md", "Title", "Chapter",
                                      "01-chapter", "01-deep-learning", "Part",
                                      "content/zh/sections/01-01.tex", "standalone/zh/01-01.tex")
            old = output / "old-folder" / "01-01.pdf"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"old")
            state = {"zh:01-01": {"status": "success", "path": str(old), "bytes": 3}}
            builder.refresh_state(state, {"zh": [section]}, output)
            self.assertEqual(state["zh:01-01"]["status"], "stale")
            self.assertIn("obsolete", state["zh:01-01"]["error"])

    def test_atomic_report_replacement_and_failure_preservation(self):
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            destination = Path(folder) / "state.json"
            builder.atomic_text(destination, '{"status": "old"}\n')
            with patch.object(builder.os, "replace", side_effect=OSError("replacement failed")):
                with self.assertRaises(OSError):
                    builder.atomic_text(destination, '{"status": "new"}\n')
            self.assertEqual(destination.read_text(encoding="utf-8"), '{"status": "old"}\n')
            self.assertEqual(list(destination.parent.iterdir()), [destination])
            builder.atomic_text(destination, '{"status": "new"}\n')
            self.assertEqual(destination.read_text(encoding="utf-8"), '{"status": "new"}\n')

    def test_output_lock_rejects_concurrent_writer_and_releases(self):
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            reports = Path(folder)
            with builder.output_lock(reports):
                with self.assertRaises(RuntimeError), builder.output_lock(reports):
                    pass
            with builder.output_lock(reports):
                pass

    def test_failed_build_preserves_previous_output(self):
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            base = Path(folder)
            destination = base / "old.pdf"
            destination.write_bytes(b"previous-success")
            with patch.object(builder, "PROJECT", base):
                result = builder.compile_pdf("en", "01-01", base / "missing.tex", destination, base / "reports")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(destination.read_bytes(), b"previous-success")
            self.assertFalse(list((base / "build" / "work").iterdir()))

    def test_snapshot_tracks_inputs_and_rejects_escape(self):
        with tempfile.TemporaryDirectory(prefix="ainote-test-") as folder:
            base = Path(folder)
            main, child = base / "main.tex", base / "child.tex"
            main.write_text(r"\input{child.tex}", encoding="utf-8")
            child.write_text("text", encoding="utf-8")
            with patch.object(builder, "PROJECT", base):
                before = builder.input_snapshot(main)
                self.assertEqual(set(before), {"main.tex", "child.tex"})
                child.write_text("changed text", encoding="utf-8")
                self.assertNotEqual(before, builder.input_snapshot(main))
                main.write_text(r"\input{../outside.tex}", encoding="utf-8")
                with self.assertRaises(ValueError):
                    builder.input_snapshot(main)


if __name__ == "__main__":
    unittest.main()
