---
title: "33.4 Modeling Touch"
chapter_title: "World–Action Models"
section_id: "33-04"
language: en
source_language: zh
source_docx: "第6部分 具身智能与世界模型/33.世界-动作模型/33.4 触觉的建模.docx"
status: "manually rebuilt and checked against Word"
ocr: "all Word-visible text and formula images manually transcribed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 33.4 Modeling Touch

## I. Challenges of Tactile Modeling

1. Lack of methods for obtaining large-scale tactile data. Effective tactile data available in industry is far scarcer than language or vision data. Tactile-equipped collection devices also retain high barriers to adoption, making collection difficult to scale.

2. Difficulty jointly modeling touch with language and vision. Touch usually occurs at the instant of robot–object contact and is high-frequency, while the signal may be absent at other times and thus sparse. Tactile data therefore cannot simply be mixed indiscriminately with language and vision.

3. Difficulty transferring across tactile sensors. Sensors include vision-based tactile, piezoresistive, capacitive, and other types, producing image, array, state, and other signal modalities at different resolutions, making general reuse difficult.

4. Different end-effector types have different sensor distributions. Grippers and dexterous hands, for example, differ in tactile-sensor placement and count, yet must operate through the same robot policy. This increases modeling difficulty, particularly for whole-body robot policies.

## II. Main Approaches to Tactile Modeling

### (1) Keeping the Tactile Model Relatively Separate from the Main Policy

This approach makes touch an independent high-frequency correction module attached outside the main policy. For example, CraftNet divides a robot into three asynchronously operating components. System0 is an interaction loop running hundreds of times per second, specifically receiving force and torque signals and adjusting fingertip positions in real time at contact. Touch serves as the fastest correction loop.

![CraftNet architecture separating the tactile model from the main policy](../../../assets/images/06-embodied-ai-world-models/33-04/craftnet-architecture.png)

The drawback of this hierarchy is that touch does not enter backbone pretraining, limiting scaling potential compared with end-to-end unification and preventing cross-embodiment reuse.

### (2) Connecting to the Main Policy through an Adapter

Tactile-VLA, for example, uses an adapter to inject tactile tokens into a vision–language model, sharing attention with vision and language rather than building a separate system. Its advantage is that visual information can enter the main policy, achieving end-to-end unification and avoiding the inherent disadvantages of hierarchical architectures.

![Tactile-VLA architecture connecting to the main policy through an adapter](../../../assets/images/06-embodied-ai-world-models/33-04/tactile-vla-architecture.png)

Two drawbacks remain:

1. When tactile information is later introduced to fine-tune an already trained vision–language model, touch and existing vision–language knowledge can interfere. Without proper modality fusion, forcibly combining touch with vision and language can not only fail to help but disrupt vision–language perception and harm overall performance.

2. The adapter remains tied to one sensor and cannot be reused after the sensor changes.

### (3) Separating Modalities with MoE inside the Main Policy

A mixture of experts can use gating to control which expert dominates at different stages, relying mainly on vision without contact and sharply increasing tactile weight during contact. ForceVLA, for example, projects force readings into tokens and uses a force-aware MoE during action decoding for modality- and stage-aware routing. ForceVLA2 feeds forces into the VLM as text prompts to form “force-aware task concepts,” while also applying cross-scale MoE on the action side.

MoE offers two benefits:

1. Because touch differs substantially from vision and language, separating them through different weights avoids damaging pretrained vision–language knowledge.

2. Once the tactile-input format is defined, shared pretrained tactile experts can be reused during fine-tuning for previously unseen sensors. Only a lightweight encoder needs training to convert sensor data into a format readable by the tactile expert.

## III. FTP-1: A Foundation Model Incorporating Tactile Modeling

### (1) Specifying the Tactile-Sensor Data Format

“FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation” proposes the MTTS data interface. Its coordinates use the hand's functional regions, dividing each hand into 24 regions. Signals from any tactile sensor are assigned to these regions and translated into the same set of slot tokens. Each token has a functional-region embedding identifying where the sensation originates.

### (2) Translating Different Sensor-Signal Types

Image-based signals (deformation maps): Train a lightweight ViT to process images and convert them into MTTS-format tokens.

Array-based signals (pressure grids): Train a lightweight CNN to extract features and convert them into MTTS-format tokens.

State-based signals (force readings): Apply Fourier encoding, then train a lightweight MLP to transform the data format into MTTS-compatible tokens.

### (3) Model Architecture

![FTP-1 tactile foundation-model architecture](../../../assets/images/06-embodied-ai-world-models/33-04/ftp1-architecture.png)

After translation, different sensor signals enter the backbone, mainly composed of vision–language, action, and tactile experts. The vision–language expert has VLM capabilities, the action expert generates continuous actions through flow matching, and the tactile expert processes tactile signals. The action expert can obtain information unidirectionally from the tactile expert.

## References

- Sharpa. (2026). [Sharpa Announces CraftNet: A Hierarchical VTLA Model for Fine Manipulation](https://www.sharpa.com/blogs/news/sharpa-announces-craftnet-a-hierarchical-vtla-model-for-fine-manipulation). Official technical article.
- Huang, J., Wang, S., Lin, F., Hu, Y., Wen, C., & Gao, Y. (2025). [Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization](https://arxiv.org/abs/2507.09160). arXiv:2507.09160.
- Yu, J., Liu, H., Yu, Q., et al. (2025). [ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](https://arxiv.org/abs/2505.22159). arXiv:2505.22159.
- Li, Y., Zhaxizhuoma, Jiang, H., et al. (2026). [ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation](https://arxiv.org/abs/2603.15169). arXiv:2603.15169.
- Yuan, C., Zhang, Z., Zhou, M., et al. (2026). [FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation](https://arxiv.org/abs/2606.13102). arXiv:2606.13102.
- 具身纪元. (2026). [Touch Needs Its Own Foundation Model and Open-X Embodiment](https://www.jintiankansha.com/t/DWG4NegaMt) (translated title; in Chinese). WeChat public-account article.
