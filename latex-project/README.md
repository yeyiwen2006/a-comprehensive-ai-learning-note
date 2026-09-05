# A Comprehensive AI Learning Note LaTeX 项目

[![简体中文（当前）](https://img.shields.io/badge/简体中文-当前-blue)](README.md) [![English](https://img.shields.io/badge/English-switch-lightgrey)](README_EN.md)

本项目将公开 Markdown 转为中英文 LaTeX。中文总书入口是 `main.tex`，英文入口是 `main-en.tex`。每一节在 `content/zh/sections/` 或 `content/en/sections/` 中只有一份正文，总书和 `standalone/` 下的单节入口共同引用它。生成器只做排版转换，不进行翻译，也不改写原文。

## 环境要求

- Python 3.10 或更新版本，安装 `Pillow` 和 `pypdf`。
- Pandoc，以及含 XeLaTeX、ctex 及模板所用宏包的 TeX Live 或 MiKTeX。
- Poppler 的 `pdfinfo`、`pdftotext`；视觉检查另外使用 `pdftoppm`。
- 字体 Times New Roman、Arial、Consolas、SimSun；如有 SimHei、KaiTi、Microsoft YaHei，也会使用。其他系统需安装相应字体，或在生成器的基础模板中配置具有所需字符覆盖的替代字体后重新生成。

## 生成和编译

以下命令从仓库根目录执行：

```powershell
# 生成两种语言的共享正文、总书入口、单节入口和资源映射
python -B -X utf8 latex-project/scripts/build_latex_project.py --languages zh,en

# 两本总书
python -B -X utf8 latex-project/scripts/build_pdfs.py --scope book --language all

# 单独编译第1.3节，可选 zh、en、all
python -B -X utf8 latex-project/scripts/build_pdfs.py --scope section --language all --section 01-03

# 全部336个单节 PDF
python -B -X utf8 latex-project/scripts/build_pdfs.py --scope sections --language all

# 两本总书和全部单节
python -B -X utf8 latex-project/scripts/build_pdfs.py --scope all --language all

# 自定义输出目录和并行数（1至8）
python -B -X utf8 latex-project/scripts/build_pdfs.py --scope all --language all --output-root "E:/AI-Note-PDFs" --jobs 3
```

生成器的 `--languages zh` 或 `--languages en` 只更新一种语言。中文版必须有168节，完整英文版必须与中文节编号一一对应。`--allow-partial` 仅用于开发预览，不代表完整版本，也不能绕过完整 PDF 构建的数量检查。请修改 Markdown 后重新生成，不直接修改生成的正文 TeX。

每个任务在独立临时目录中运行三遍 XeLaTeX。新 PDF 通过日志、页数、可解析性、作者、语言和文本检查后，才替换正式文件。失败不会覆盖上一次成功的 PDF；其他任务可以继续，但整次构建最终返回失败。`--jobs` 控制同一命令内部的并行任务数；输出锁会拒绝多个命令同时写入同一输出目录。

## PDF 输出位置

仓库位于 `github-export/<repo>/` 时，默认输出到其所在工作区根目录的 `output/pdf/`；独立克隆时，默认输出到仓库内的 `output/pdf/`。可用 `--output-root` 覆盖。构建开始和结束都会显示解析后的绝对路径。

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

`PDF-INDEX.md` 用相对链接列出两本总书和全部单节的标题、部分、章节、页数及状态。整个 `output/pdf/` 目录移动后，PDF 链接仍可使用。生成的 PDF、报告与索引均为本地文件，不上传到仓库。

两本总书同时保留在原 `latex-project/build/` 和 `output/pdf/complete-books/`，文件名分别固定为：

- `a-comprehensive-ai-learning-note（位于github同名仓库）.pdf`
- `a-comprehensive-ai-learning-note-en.pdf`

单节 PDF 仅存入语言对应目录，文件名以稳定编号开头、以 `-ZH.pdf` 或 `-EN.pdf` 结尾，不在 `latex-project/build/` 另存一套。

## 直接使用 XeLaTeX

以下命令在 `latex-project/` 目录执行。每种语言连续编译三次，使目录和引用稳定：

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

这种方式只生成 `build/` 内的 PDF，不更新集中输出、索引或报告。需要双位置总书、统一检查和索引时，请使用上面的 Python 构建命令。

## 验证与维护

```powershell
python -B -X utf8 latex-project/scripts/test_bilingual_build.py
python -B -X utf8 tools/validate_generated_markdown.py
python -B -X utf8 tools/content_statistics.py
```

英文译文或标题修改后，可运行 `python -B -X utf8 tools/update_english_navigation.py`，按中文目录与学习路径的原有顺序更新英文入口；该命令不翻译正文。统计脚本会分别报告中文正文计数、英文词数、节数和图片资源数；可选的 `--word-root` 只读取本地 Word 文档的媒体与引用元数据，不提取或公开原图。

`resource-map.json` 维护稳定的图片映射，两种语言的资源清单分别为 `RESOURCE_MANIFEST.md` 和 `RESOURCE_MANIFEST_EN.md`。自动检查不能替代逐段翻译复核及 PDF 视觉检查。发布前还需检查两本总书的所有页面、各单节的首尾页及复杂页面，并确认中文、英文 Markdown 和 LaTeX 同步。构建报告只记录本地自动检查结果，不代表已完成视觉验收或 GitHub/Gitee 同步。不进行文件哈希校验。
