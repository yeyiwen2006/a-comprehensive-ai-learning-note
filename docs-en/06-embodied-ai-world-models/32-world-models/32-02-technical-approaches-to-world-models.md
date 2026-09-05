---
title: "32.2 Technical Approaches to World Models"
chapter_title: "World Models"
section_id: "32-02"
language: en
source_language: zh
source_docx: "第6部分 具身智能与世界模型/32.世界模型/32.2 世界模型的技术路线.docx"
status: "manually rebuilt and checked against Word"
ocr: "all Word-visible text and formula images manually transcribed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 32.2 Technical Approaches to World Models

## I. Main Technical Approaches to World Models

### (1) Latent-Variable World Models

Latent-variable world models are among the earliest forms of world models, including RSSM and related derivatives. They compress the world state at time t into a latent z\_t and combine it with action a\_t and other information to predict the next latent z\_t+1. They are generally trained using or incorporating downstream-task losses.

This approach rarely appears on its own in industry, but it has not disappeared; instead, it has been incorporated into generative world models. Video pixels have very high dimensionality, but according to the manifold hypothesis, meaningful data actually lies in lower dimensions. Inputs are therefore generally first compressed into low-dimensional latents for processing in latent space, then restored to high dimensionality for output.

### (2) Generative World Models

Generative world models are currently the mainstream form. By generating future video frames and other outputs, they learn autonomously from videos and similar data. Their training can broadly follow that of multimodal generative models. Current generative world models are almost all combined with the latent-variable approach.

1. Loss computation: The final generated high-dimensional frames can be used directly to compute the loss. Alternatively, a VAE or another autoencoder can first be trained and frozen, followed by pretraining of a generative model in latent space using latent flow-matching or similar losses directly. Other downstream-task losses can also be used for training.

2. Generation paradigms

(1) Autoregression: Inherits LLMs' next-token-prediction idea to predict the next state token. Its disadvantages are that high-dimensional visual generation is usually slow and errors can progressively accumulate.

(2) Diffusion: Starts from noise and gradually denoises future videos or states. Outputs become increasingly clear from an initially blurred state, with information content growing progressively, resembling human perception or imagination progressing from rough outlines to concrete details. Its disadvantages are high GPU-memory use and computational cost in high dimensions, so it is generally used only to generate latents in low-dimensional spaces.

### (3) Viewpoint-Modeling World Models

This approach treats each 2D frame as a different observation of the same 3D world. Given user-provided images and their viewpoints, it implicitly maintains a multiview 3D world model and outputs the corresponding 2D representation from the user's viewpoint. Representative work includes RTFM and IC-World.

1. Advantages: Modeling 3D space is consistent with the binocular/multi-eye views of humans and animals and the consistency required across multiple robot sensors. It may be key to spatial understanding and an important technical foundation for embodied AI.

2. Disadvantages: It must handle viewpoints and correctly determine their transformations, but training videos do not inherently include viewpoint information, making it difficult to recognize the shared 3D identity of multiple views at test time. Adding estimated viewpoints to training data through hard-coded algorithms introduces errors; performance may be difficult to maintain as scenes become more complex, making data-driven scaling difficult.

### (4) Object-Modeling World Models

This approach separates, extracts, and explicitly models individual objects in space, their properties, and their relationships. Rather than viewing the world as a single image, it views it as cups, tables, and robotic arms, together with their positions, masses, contact relationships, friction, and so on. The model contains a fixed number of “slots,” into which object information enters competitively, enabling important objects' information to be handled during subsequent generation. Representative work includes C-SWM, SlotFormer, and Marble.

1. Advantages: Well suited to spatial relationships, contact, occlusion, and compositional generalization. It is highly interpretable, interventions are simple (such as adding or moving objects), and it is closer to genuine physical reasoning.

2. Disadvantages: It must process complex visual information and decompose the world into the correct structure, requiring training on structured data, which is extremely scarce. Decomposition itself is also error-prone, and errors at this step cause subsequent prediction errors.

### (5) World Models That Predict in Abstract Representation Spaces

Predicting every pixel is unnecessary in everyday life and natural science. For example, object-motion prediction only requires abstractions such as point masses and rigid bodies, not pixel- or atom-level analysis. Following this idea, we map observations into hidden states as in generative world models, but no longer drive learning with direct or indirect pixel objectives such as pixel generation or reconstruction. Instead, the model learns hidden-state representations autonomously through self-supervised prediction, using only regularization losses to avoid representation collapse. One current method extracts all hidden states from observations, randomly masks them in space and time, and trains the model to predict the hidden states at masked positions. By minimizing the gap between predicted and actual hidden states in latent space, it learns spatiotemporal and physical representations. Representative work includes the V-JEPA series and LeWorldModel.

1. Advantages: Can naturally encode vision, touch, sensor data, and other modalities together, and best matches the thinking process of scientific reasoning. It does not need to predict pixels, saving compute and enabling high speed.

2. Disadvantages: Abstract representation spaces are invisible to humans, making problems difficult to identify directly. The model is essentially an encoder, requiring fine-tuning for downstream tasks.

## II. How Is Action Conditioning Injected into a Video World Model?

Many world models are fine-tuned from video-generation models. A video world model does not merely predict the next frame from existing frames, but predicts it from existing frames and action interventions. How action interventions are injected into the video-generation backbone is therefore crucial. An action-injection head is generally designed, with typical injection methods including:

1. Inject actions through cross-attention in the DiT backbone.

2. Use action-injection values to control AdaLN parameters.

Fine-tuning follows injection. Initially, the video-generation backbone is generally frozen and only the action-injection head is tuned; some models later undergo full fine-tuning.

## III. Learning an Action Space: Latent Action Models

The core expression of a world model is s\_t+1=f(s\_t,a\_t). In vast quantities of video data, s\_t and s\_t+1 are easy to obtain, but explicit action data a\_t is unavailable. This may affect learning of the consequences of specific interventions, particularly in interactive or embodied settings. If actions are regarded as a modality, we may need a corresponding “vocabulary,” namely an action space. Latent action models learn such an action space from videos through self-supervision.

### (1) Core Components

1. Inverse dynamics model: Infer a latent representation of action a\_t from preceding and subsequent states s\_t and s\_t+1, using an information bottleneck to encourage a\_t to encode the most important motion information.

2. Forward dynamics model: Use the current state s\_t and action a\_t to predict the next state s\_t+1.

### (2) Abstraction Requirements

If action a\_t is allowed to retain too much information to improve generation quality, content such as colors, textures, and backgrounds can easily enter the representation, making actions insufficiently abstract and difficult to transfer. To ensure transferable and abstract action representations, models typically impose a strong prediction bottleneck, such as vector quantization or a variational bottleneck.

However, sufficiently abstract compression undeniably makes transfer easier while potentially departing from the action space's intrinsic manifold structure, sacrificing next-state prediction accuracy and the details required for generation.

## References

- Hafner, D., Lillicrap, T., Fischer, I., et al. (2019). [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551). ICML. (PlaNet; the paper introduces RSSM)
- World Labs. (2025). [RTFM: A Real-Time Frame Model](https://www.worldlabs.ai/blog/rtfm). Official research article.
- Wu, F., Wei, J., Li, R., et al. (2025). [IC-World: In-Context Generation for Shared World Modeling](https://arxiv.org/abs/2512.02793). arXiv:2512.02793.
- Kipf, T., van der Pol, E., & Welling, M. (2019). [Contrastive Learning of Structured World Models](https://arxiv.org/abs/1911.12247). arXiv:1911.12247.
- Wu, Z., Dvornik, N., Greff, K., Kipf, T., & Garg, A. (2023). [SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models](https://arxiv.org/abs/2210.05861). ICLR 2023.
- World Labs. (2025). [Marble: A Multimodal World Model](https://www.worldlabs.ai/blog/marble-world-model). Official research article.
- Bardes, A., Garrido, Q., Ponce, J., et al. (2024). [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471). arXiv:2404.08471. (V-JEPA)
- Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026). [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312). arXiv:2603.19312.
- Garrido, Q., Nagarajan, T., Terver, B., Ballas, N., LeCun, Y., & Rabbat, M. (2026). [Learning Latent Action World Models In The Wild](https://arxiv.org/abs/2601.05230). arXiv:2601.05230.
