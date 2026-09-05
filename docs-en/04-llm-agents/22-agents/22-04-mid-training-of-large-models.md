---
title: "22.4 Mid-Training of Large Models"
chapter_title: "Agents"
section_id: "22-04"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/22.智能体/22.4 大模型的中期训练.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "not applicable; source contains no Word images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 22.4 Mid-Training of Large Models

## I. Background

In traditional single-turn dialogue or reasoning tasks, models need depth in one capability, such as maximizing reasoning ability or making dialogue fluent. In agent settings, however, they may need to connect retrieval, code diagnosis, testing, tool use, context management, and other capabilities into a closed loop of planning, reasoning, execution, and feedback. Some specialized domains may also require more corpora for specialized training. Current post-training mainly optimizes performance within a pretrained model's existing capability boundaries, making it difficult to create entirely unseen capabilities or train behaviors never previously learned, such as "exception handling after tool calls." Mid-training can therefore be introduced before post-training to inject the required behavioral priors.

## II. Data Types

High-quality long documents, domain corpora, code repositories (such as GitHub PRs), mathematical reasoning, agent trajectories, and long-context data. For a coding agent, for example, agents can autonomously perform different real tasks in different Docker environments while recording the complete interaction with the environment. Every action and every observation returned by the environment becomes training data.

## III. Training Method

1. Mid-training uses standard next-token prediction. Its objective is to shift the model's capability distribution at the knowledge level through domain-specific data, rather than teach specific instruction-following behavior as SFT does.

2. No loss mask is used. In SFT or RL, a loss mask is needed because we optimize the model's output policy, whereas user prompts and environmental feedback cannot be optimized and should not be targets of optimization during training. A loss mask prevents them from interfering. Mid-training instead focuses on "learning knowledge," not "solving real problems." By predicting environmental-feedback tokens and optimizing the corresponding loss, the model learns the actual probability distribution of environmental data. It must learn not only "how to modify code" but also "how to describe a bug" and "what a codebase usually looks like."

## References

- Chen, M., Zhang, L., Feng, Y., et al. (2026). [SWE-Universe: Scale Real-World Verifiable Environments to Millions](https://arxiv.org/abs/2602.02361). arXiv:2602.02361.
- GLM-4.5 Team. (2025). [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471). arXiv:2508.06471.
- Tu, C., Zhang, X., Weng, R., et al. (2025). [A Survey on LLM Mid-training](https://arxiv.org/abs/2510.23081). arXiv:2510.23081.
- Zeng, J., Fu, D., Mi, T., et al. (2026). [daVinci-Dev: Agent-native Mid-training for Software Engineering](https://arxiv.org/abs/2601.18418). arXiv:2601.18418.
- Zhu, Q., Chen, T., Lu, S., et al. (2026). [Pull Requests as a Training Signal for Repo-Level Code Editing](https://arxiv.org/abs/2602.07457). arXiv:2602.07457.
- Zhang, C., Neubig, G., & Yue, X. (2025). [On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models](https://arxiv.org/abs/2512.07783). arXiv:2512.07783.
- Wang, Z., Zhou, F., Li, X., & Liu, P. (2025). [OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling](https://arxiv.org/abs/2506.20512). arXiv:2506.20512.
