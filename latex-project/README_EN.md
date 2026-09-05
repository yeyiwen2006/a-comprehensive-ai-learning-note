# A Comprehensive AI Learning Note LaTeX Project

[![简体中文](https://img.shields.io/badge/简体中文-switch-lightgrey)](README.md) [![English (Current)](https://img.shields.io/badge/English-Current-blue)](README_EN.md)

This project converts the public Markdown into Chinese and English LaTeX. The Chinese book entry point is `main.tex`, and the English entry point is `main-en.tex`. Each section has a single body file under `content/zh/sections/` or `content/en/sections/`, shared by the book and the section entry under `standalone/`. The generator performs typesetting conversion only; it neither translates nor rewrites the source.

## Requirements

- Python 3.10 or later, with `Pillow` and `pypdf` installed.
- Pandoc, and TeX Live or MiKTeX with XeLaTeX, ctex, and the packages used by the template.
- Poppler's `pdfinfo` and `pdftotext`; visual inspection additionally uses `pdftoppm`.
- Fonts: Times New Roman, Arial, Consolas, and SimSun. SimHei, KaiTi, and Microsoft YaHei are also used when available. On other systems, install these fonts or configure replacements with the required character coverage in the generator's base template, then regenerate.

## Generation and Compilation

Run the following commands from the repository root:

```powershell
# Generate shared content, book entries, section entries, and image mappings for both languages
python -B -X utf8 latex-project/scripts/build_latex_project.py --languages zh,en

# Both complete books
python -B -X utf8 latex-project/scripts/build_pdfs.py --scope book --language all

# Section 1.3 only; choose zh, en, or all
python -B -X utf8 latex-project/scripts/build_pdfs.py --scope section --language all --section 01-03

# All 336 section PDFs
python -B -X utf8 latex-project/scripts/build_pdfs.py --scope sections --language all

# Both books and all sections
python -B -X utf8 latex-project/scripts/build_pdfs.py --scope all --language all

# Custom output directory and concurrency (1 to 8)
python -B -X utf8 latex-project/scripts/build_pdfs.py --scope all --language all --output-root "E:/AI-Note-PDFs" --jobs 3
```

The generator's `--languages zh` or `--languages en` updates one language only. The Chinese edition must contain 168 sections, and the complete English edition must match the Chinese section IDs one-to-one. `--allow-partial` is for development previews only. It does not indicate a complete edition or bypass the count checks for complete PDF builds. Edit Markdown and regenerate rather than editing the generated body TeX directly.

Each job runs XeLaTeX three times in its own temporary directory. A new PDF replaces the final local file only after checks of its log, page count, parseability, author, language, and text pass. A failure does not overwrite the previous successful PDF. Other jobs may continue, but the overall command returns a failure status. `--jobs` controls concurrency within one command; the output lock rejects multiple commands writing to the same output directory at once.

## PDF Output Locations

When the repository is under `github-export/<repo>/`, the default output is `output/pdf/` at the enclosing workspace root. For a standalone clone, it is `output/pdf/` inside the repository. Override this with `--output-root`. The resolved absolute path is printed at the start and end of the build.

```text
output/pdf/
├── PDF-INDEX.md
├── complete-books/
├── chinese-sections/
│   └── part-01-deep-learning/chapter-01-深度学习基础理论/...
├── english-sections/
│   └── part-01-deep-learning/chapter-01-fundamentals-of-deep-learning/...
└── build-reports/
    ├── BUILD-REPORT.md
    ├── build-state.json
    ├── failed-sections.txt
    └── logs/
```

`PDF-INDEX.md` uses relative links to list the two books and every section, including titles, parts, chapters, page counts, and status. PDF links remain usable if the entire `output/pdf/` directory is moved. Generated PDFs, reports, and the index are local files and are not uploaded to the repository.

Both books are retained in the original `latex-project/build/` and in `output/pdf/complete-books/`, with these fixed filenames:

- `a-comprehensive-ai-learning-note（位于github同名仓库）.pdf`
- `a-comprehensive-ai-learning-note-en.pdf`

Section PDFs are stored only in their language-specific directories. Their filenames start with a stable section ID and end in `-ZH.pdf` or `-EN.pdf`; no duplicate set is kept in `latex-project/build/`.

## Direct XeLaTeX Compilation

Run the following in `latex-project/`. Compile each language three consecutive times to stabilize the table of contents and references:

```powershell
New-Item -ItemType Directory -Force build | Out-Null
1..3 | ForEach-Object {
    xelatex -interaction=nonstopmode -halt-on-error -jobname="a-comprehensive-ai-learning-note（位于github同名仓库）" -output-directory=build main.tex
    if ($LASTEXITCODE -ne 0) { throw "Chinese PDF compilation failed" }
}
1..3 | ForEach-Object {
    xelatex -interaction=nonstopmode -halt-on-error -jobname="a-comprehensive-ai-learning-note-en" -output-directory=build main-en.tex
    if ($LASTEXITCODE -ne 0) { throw "English PDF compilation failed" }
}
```

This approach creates PDFs in `build/` only; it does not update the consolidated output, index, or reports. Use the Python build commands above for books in both locations, unified checks, and an index.

## Validation and Maintenance

```powershell
python -B -X utf8 latex-project/scripts/test_bilingual_build.py
python -B -X utf8 tools/validate_generated_markdown.py
python -B -X utf8 tools/content_statistics.py
```

After changing English translations or titles, run `python -B -X utf8 tools/update_english_navigation.py` to update the English navigation in the original order of the Chinese table of contents and learning path. This command does not translate body text. The statistics script reports the Chinese body-text count, English word count, section counts, and image-resource counts separately. The optional `--word-root` reads only media and reference metadata from local Word documents; it does not extract or publish original images.

`resource-map.json` maintains stable image mappings. The language-specific resource manifests are `RESOURCE_MANIFEST.md` and `RESOURCE_MANIFEST_EN.md`. Automated checks do not replace paragraph-by-paragraph translation review or visual PDF inspection. Before publication, inspect every page of both books, the first and last pages of each section, and complex pages, and confirm that Chinese and English Markdown and LaTeX remain synchronized. The build report records local automated checks only; it does not imply completed visual acceptance or GitHub/Gitee synchronization. No file hashes are computed.
