---
title: "5.6 The Muon Optimizer"
chapter_title: "Optimization Algorithms"
section_id: "05-06"
language: en
source_language: zh
source_docx: "第1部分 深度学习/5.优化算法/5.6 Muon优化器.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 5.6 The Muon Optimizer

Adam/AdamW implements adaptive learning rates by scaling each dimension (parameter) separately through division by a scalar. In optimization, however, these dimensions represent only one coordinate system in a high-dimensional parameter space. Muon updates parameters by orthogonalizing the momentum matrix, making update matrix $U$ approximately orthogonal, satisfying $U^{T}U\approx I$.

Geometrically, it normalizes along all principal directions (singular-vector directions) in parameter space, taking equal-length steps along them. This resembles Newton's method performing gradient descent with equal step lengths in each direction of a transformed space (although these singular-vector directions are not Newton's directions), rather than Adam's attempt merely to adjust movement along individual coordinate axes.

For matrix orthogonalization, Muon uses Newton–Schulz iteration to approximate the matrix "sign function," greatly reducing computation.

Some large-model training now uses modified Muon variants. Kimi K2, for example, uses MuonClip, which adds QK-clip to Muon, rather than the unmodified original optimizer.

## References

- Jordan, K., et al. (2024). [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/).
- Kimi Team. (2025). [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534). arXiv:2507.20534.
