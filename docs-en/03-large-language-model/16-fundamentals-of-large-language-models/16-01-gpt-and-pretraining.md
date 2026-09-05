---
title: "16.1 GPT and Pretraining"
chapter_title: "Fundamentals of Large Language Models"
section_id: "16-01"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/16.大语言模型的基本原理/16.1 GPT与预训练.docx"
status: "synced-from-docx"
ocr: "disabled; no images in source"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 16.1 GPT and Pretraining

## I. Core Idea

The central idea of the Generative Pre-trained Transformer (GPT) is “predicting the next token,” a unidirectional, autoregressive generation process. For example, the first step predicts the first token (denoted $x_1$) from the existing prompt; the second predicts $x_2$ from the prompt and $x_1$; the third predicts $x_3$ from the prompt, $x_1$, and $x_2$; and so on. The underlying logic is that a truly powerful language model can learn almost all the regularities of language, knowledge, reasoning, and thought through the task of “predicting the next word” alone. When the model's scale (parameters, data, and compute) is sufficiently large, this seemingly simple task forces it to construct an extremely complex and precise model of the world to make better predictions. This training process is called “pretraining” to distinguish it from subsequent fine-tuning (post-training).

In effect, training such a model “compresses” a rich understanding of the world into its parameters. One view holds that “compression is intelligence”: to reconstruct information with high confidence during generation, the model must distill underlying regularities from vast amounts of knowledge, approaching “lossless compression” as closely as possible during training. This distillation process can also reveal underlying connections between knowledge in different fields that humans may not discover. This is one source of innovation (so we cannot categorically claim that AI cannot innovate). Reinforcement fine-tuning, discussed later, is also an important reason why AI can develop new strategies.

A specific clarification: GPT here denotes the general architecture of a “generative pre-trained Transformer,” not specifically OpenAI's ChatGPT model family (although ChatGPT, as the first generative AI product, was presumably named after this architecture).

## II. Model Architecture

Like BERT, GPT is based on the Transformer. Unlike BERT, however, GPT selects the decoder portion of the architecture and, among the three attention blocks, uses only masked multi-head self-attention.

In this attention block, queries, keys, and values all come from the already generated part of the same sequence. Each word can attend only to words on its left (past words), not those on its right (future words), determining GPT's “unidirectionality.” An attention mask hides all future-position information, ensuring that the model cannot “peek” at future answers while generating the next word. This matches its behavior during inference (generation mode).

A GPT model stacks multiple Transformer decoder blocks. GPT-3, for example, has 96 layers. Each decoder block also contains a fully connected feed-forward network that applies a nonlinear transformation to the representation at each position.

Why choose a decoder? Its autoregressive nature naturally fits the “next-word prediction” pretraining task. Training and inference behave identically: both predict unknown text from known text, avoiding BERT's pretraining–fine-tuning mismatch.

## III. Why Is an Encoder Unnecessary, and Why Does This Work Better than Encoder-Based BERT?

### 1. Generation Is a More Fundamental and General Capability

Understanding is a subset of generation: to generate high-quality text, a model must deeply understand language, knowledge, logic, and context. A model that converses fluently necessarily has substantial understanding. Conversely, a model that only excels at understanding, such as BERT, cannot generate fluent text. Through the “generation” paradigm, GPT can be guided to perform almost any NLP task, a process called “prompt engineering.” This unification allows one model to handle a wide variety of tasks, whereas BERT usually requires a task-specific architecture and fine-tuning procedure.

### 2. Scaling Laws

Many experiments have found that, once model size, data volume, and compute reach a sufficient scale, decoder-only Transformers exhibit surprising new capabilities such as reasoning, code generation, and complex instruction following.

At the scale of hundreds of billions of parameters (such as GPT-3), decoder-only architectures develop strong in-context learning. With just a prompt, the model can understand and perform various complex tasks without a dedicated encoder to parse the instructions.

This ultimately demonstrates that a sufficiently large autoregressive model's decoder already possesses the “deep understanding” expected of an encoder. By repeatedly processing the preceding text during generation, it can likewise achieve deep semantic understanding.

### 3. Consistency Between Pretraining and Use

GPT is trained on next-word prediction (autoregression), and its inference/use also generates a next word, appends it to the input, and continues generating the following word (autoregression). Its behavior is identical in training and deployment. It never sees the entire sentence during training (future words are masked), and does not need future words during inference. This consistency allows all capabilities learned in training to be applied to actual generation without loss.

By contrast, if GPT used an encoder with bidirectional attention to understand inputs during pretraining, generation would encounter a paradox: to generate the $N$-th word, the encoder would need the complete input sequence, but that sequence would include future words not yet generated. Discarding the encoder is therefore inevitable if maximal task consistency is to be maintained.

The mismatch between training and generation is also considered one of the bottlenecks limiting BERT's performance, especially its generation performance.

### 4. Encoder Redundancy

Encoder–decoder architectures are designed for sequence-to-sequence tasks such as machine translation and text summarization. In these tasks, inputs and outputs can have different lengths and modalities. The encoder understands and encodes the entire input bidirectionally, and the decoder then generates the output autoregressively from the encoder's output. GPT's task is closer to “text continuation” than “text transformation.”

In terms of parameter and computational efficiency, training a single decoder stack rather than both an encoder and a decoder allows each component to be “deeper” or “wider” at the same parameter count, potentially producing a more capable model. Training and inference are also simpler, avoiding complex encoder–decoder interactions.

An analogy: an encoder–decoder model such as T5 resembles a translation team. The encoder is a researcher analyzing the source language (fully understanding the original), and the decoder is a writer producing the target language from the researcher's report (generating the translation). A decoder-only model such as GPT resembles a writer working alone. The writer reads what has already been written (itself a process of “understanding”) while continuing the text. No separate researcher is needed to explain the preceding text, because the writer authored it.

### 5. The Sequential Nature of Text

At very large model scales, encoder architectures such as BERT may be less effective than autoregressive decoders because they disrupt text's natural sequential order.

## IV. Scaling Laws

Scaling laws were introduced by OpenAI in the 2020 paper *Scaling Laws for Neural Language Models* (one account suggests that Dario Amodei and colleagues had already observed an early form while at Baidu). They are empirical regularities observed in large-model research: as model size, training-data volume, and training compute increase, performance generally improves along relatively stable, predictable trends. “Performance” here is often measured by training or validation loss, whose decline approximately follows a power law, for example:

$$
\mathrm{Loss} \approx A \times N^{-\alpha}
$$

Here, $N$ can denote parameter count, data volume, or compute, while $\alpha$ describes how quickly performance improves with scale. Intuitively, larger models, more data, and greater compute generally yield lower loss and often stronger capabilities. However, these improvements are not linear; they exhibit diminishing marginal returns. More resources remain useful, but each doubling produces progressively smaller gains.

For large language models, scaling laws chiefly help researchers predict whether “a larger model will be better.” If experiments with small and medium-sized models reveal stable trends, researchers can estimate the performance of a larger model trained on more data with more compute. Training extremely large models thus need not rely entirely on blind experimentation: parameter counts, dataset scale, and training compute can be balanced in advance.

However, scaling laws do not simply mean “larger models are better.” Later research found that parameter count and training-token count must be matched appropriately. A large model with insufficient data is undertrained; abundant data paired with too small a model can encounter insufficient capacity. Chinchilla-related research, for example, emphasizes that under a fixed compute budget, many early large models had too many parameters and too little training data. A relatively smaller model exposed to more data may be preferable. Scaling laws also primarily reflect empirical observations and are not guaranteed to hold strictly across all tasks, architectures, or data-quality conditions. Changes in data quality, training methods, architectures, or task types can make results deviate from predictions.

In recent years, scaling has extended to post-training and inference. Post-training represented by RLVR follows similar scaling laws: performance continues to improve as parameter counts, training environments, and compute expand. Even with fixed parameters, increasing the computation allocated to an answer—for example, generating longer reasoning, trying multiple paths, and self-checking—can substantially improve complex-task performance, exhibiting inference-time scaling laws. Improvements in large models are therefore no longer driven only by pretraining scale, but jointly by pretraining scaling, post-training scaling, and inference-time scaling.

## V. Multimodality

Multimodal technology turns AI from a mere “text processor” into a versatile perceptual system with “eyes, ears, mouth, and brain.” It breaks down barriers between text, images, audio, and video, enabling the system to understand the world by viewing images, hearing sounds, and reading text simultaneously, as humans do. For example, besides asking questions in text, a user can directly upload a video or chart for AI to analyze its logic.

The mainstream approach currently “aligns” information from different modalities into a shared high-dimensional semantic space for processing, integrating multimodal data during pretraining. Whether the input is text, images, or audio, different processing methods ultimately bring it into the same high-dimensional semantic space. For example, the word “cat” and an image of a cat lie close together in this space, and far from an image of a watermelon. Specific implementations are introduced in the multimodal section.

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). OpenAI.
- Radford, A., Wu, J., Child, R., et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). OpenAI.
- Brown, T. B., Mann, B., Ryder, N., et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). NeurIPS 2020.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). arXiv:2001.08361.
- Amodei, D., Anubhai, R., Battenberg, E., et al. (2016). [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595). ICML 2016.
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). NeurIPS 2022.
- DeepSeek-AI, Guo, D., Yang, D., et al. (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314). arXiv:2408.03314.
