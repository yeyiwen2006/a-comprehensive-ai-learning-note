# A Comprehensive AI Learning Note (v2.0)

<!-- language-switch:start -->
[![简体中文](https://img.shields.io/badge/简体中文-switch-lightgrey)](README.md) [![English (Current)](https://img.shields.io/badge/English-Current-blue)](README_EN.md)
<!-- language-switch:end -->

This is a set of learning notes and lecture notes for learners of artificial intelligence. It covers deep learning, reinforcement learning, large language models, LLM agents, diffusion models, multimodal generation, world models, embodied AI, and related areas. Its coverage is broader than that of classic textbooks and most available resources, with detailed, accessible explanations. The first edition was released on May 10, 2026. The current edition, v2.0, contains 35 chapters and 168 sections. The Chinese body text totals approximately 374,000 counting units, excluding references and counting each Chinese character and each English or numeric word as one unit. There is no need to read everything: use the table of contents and beginner learning path under Recommended Starting Points below to read the topics that interest you.

This repository was converted from the author's local Word notes. It contains only the searchable, readable Markdown version, which supports collaborative corrections, and its corresponding LaTeX version. Chinese is the default version; the English edition is translated strictly from the Chinese edition, and the author's English name is Yiwen Ye.

The English body text contains approximately 187,000 words, excluding references, metadata, code, images, equations, and link destinations. The Chinese body text continues to use the existing counting method based on Chinese characters and English or numeric words. Reproducible counts for both languages are available through `tools/content_statistics.py`.

## Purpose of Sharing

These materials began as personal notes. After revision and expansion, they have become lecture notes that learners can consult. I am sharing them because, although existing textbooks are classics, they generally have two problems:

1. They are too concise. This can make them difficult for AI beginners to understand, while readers with some background may struggle to gain a deeper understanding. When I first studied these topics, I learned by discussing them with AI and cross-checking different resources. I have also included these thought processes in these materials.

2. They have not kept up with developments. AI is advancing rapidly, and many new areas have almost no suitable learning materials. These notes aim to help fill that gap.

These lecture notes seek to address these problems by providing as much detail as possible and covering new directions and areas in AI.

## Coverage

- Part 1: Deep Learning (9 chapters, 42 sections; approximately 78,000 counting units in the Chinese body text)
- Part 2: Reinforcement Learning (5 chapters, 25 sections; approximately 54,000 counting units in the Chinese body text)
- Part 3: Large Language Models (7 chapters, 47 sections; approximately 98,000 counting units in the Chinese body text)
- Part 4: LLM Agents (4 chapters, 14 sections; approximately 29,000 counting units in the Chinese body text)
- Part 5: Diffusion Models and Multimodal Generation (4 chapters, 22 sections; approximately 67,000 counting units in the Chinese body text)
- Part 6: Embodied AI and World Models (6 chapters, 18 sections; approximately 48,000 counting units in the Chinese body text)

## Version History

- v1.0 was released on May 10, 2026.
- v1.1 was released on May 22, 2026.

  Changes: additions to some text and figures.
- v2.0 was released on September 2, 2026 (current version).

  Changes: minor adjustments to Part 2 (Reinforcement Learning) and Part 3 (Large Language Models), removing some topics that were less mainstream or had limited learning value. The former Parts 4 and 5 (covering LLM agents, diffusion models, multimodal generation, embodied AI, world models, and related topics) were expanded, reduced, and revised, and reorganized into the current Part 4 (LLM Agents), Part 5 (Diffusion Models and Multimodal Generation), and Part 6 (Embodied AI and World Models). These revisions added new directions and trends in the field, removed some less mainstream topics or those with limited learning value, and revised some wording. `README.md` and the beginner learning path recommendations were also revised.

## Recommended Starting Points

- [Beginner Learning Path](BEGINNER_LEARNING_PATH_EN.md)
- [Full Table of Contents](TABLE_OF_CONTENTS_EN.md)
- [LaTeX Version](latex-project/README_EN.md)
- [Disclaimer](DISCLAIMER_EN.md)
- [Contributing](CONTRIBUTING_EN.md)

## LaTeX Version

This repository also provides Chinese and English LaTeX typesetting projects in [latex-project](latex-project/README_EN.md). They are generated from the public Markdown content. The Chinese book entry point is `latex-project/main.tex`, and the English book entry point is `latex-project/main-en.tex`. Individual Chinese or English sections can also be compiled by section ID. Markdown remains the primary format for reading and maintenance; if the versions differ, the Markdown documents take precedence.

Locally built PDFs are collected under `output/pdf/`: `complete-books/` holds the two books, `chinese-sections/` and `english-sections/` each hold 168 section PDFs, `build-reports/` holds build reports, and `PDF-INDEX.md` provides clickable links to the books and every section. Both complete PDFs are also retained in `latex-project/build/`. PDFs and intermediate build files are not uploaded to the repository. See the [build instructions](latex-project/README_EN.md) for commands and output-location rules.

## Contributions and Maintenance

Issues and pull requests reporting problems with concepts, formulas, references, image transcription, or typesetting are welcome. The author is currently a student and may not be able to review or merge pull requests promptly. Thank you for your understanding and support!

## Sources and Reliability

These lecture notes began as personal study notes. They include material organized by the author, notes from textbooks and papers, and summaries of developments in the field. They cite some papers, technical reports, and other developments published by academic and industry researchers in venues and platforms such as Nature, AAAI, ICML, ICLR, NeurIPS, and arXiv, but some sources may not have been cited. Please treat the original papers, official documentation, and authoritative textbooks as the final reference.

Some content reflects personal interpretations. Corrections are welcome.

Some material comes from frontier research papers and represents the methods in those papers or the author's understanding. It should not be treated directly as field-wide consensus, engineering best practice, or an industry-standard approach.

## License

This repository is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You may copy, share, and adapt the project content with attribution, for noncommercial purposes, and under the same license. The full legal terms are set out in `LICENSE` and on the official Creative Commons page.
