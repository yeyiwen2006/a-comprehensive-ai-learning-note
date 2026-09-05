---
title: "34.4 Data Processing and Alignment"
chapter_title: "Data Paradigms for Embodied AI"
section_id: "34-04"
language: en
source_language: zh
source_docx: "第6部分 具身智能与世界模型/34.具身智能的数据范式/34.4 数据的处理与对齐.docx"
status: "manually rebuilt and checked against Word"
ocr: "all Word-visible text and formula images manually transcribed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 34.4 Data Processing and Alignment

Beyond scale and diversity, how embodied AI data is processed and aligned with training needs deserves close attention.

## I. π0.7: Processing Data through Prompts

The key to π0.7's emergent compositional generalization is processing diverse data with diverse prompts. Earlier VLA training provided only “clean the refrigerator,” yielding a single signal. π0.7 expands the prompt into four layers:

1. Task instructions, such as cleaning the kitchen.

2. Subtask instructions, such as opening the refrigerator, taking out ingredients, and cleaning the tabletop.

3. Subgoal images, specifying “what the next frame should look like.”

4. Episode metadata, such as data-quality scores, whether errors occurred, and execution speed.

This rich context lets the model distinguish good from bad, fast from slow, and correct from incorrect training data, making previously unusable data useful: failed rollouts, low-quality demonstrations, and fragments from other robots. The added prompt layer tells the model “what quality this data has and which strategy produced it.”

## II. Qwen-RobotManip: Data Alignment

Data from many robots and datasets cannot simply be gathered together. The same action, “push the cup to the edge of the table,” may be recorded very differently. One robot records how much each of seven joints rotated, another records end-effector position and orientation in its base coordinates, and a dexterous hand may have more than twenty independently moving joints. Although these numbers describe similar actions, they are not the same “language.” Mixing them directly makes it difficult for the model to recognize equivalent changes, often causing it to learn noise. Qwen-RobotManip aligns and cleans data at multiple levels and observes steadily improving performance as data grows.

### (1) Format Alignment across Data Sources

#### 1. Aligning Action-Dimension Definitions

Robot structures differ greatly: single or dual arms, ordinary grippers or dexterous hands with a dozen or even dozens of degrees of freedom. Their original state and action dimensions differ completely. Qwen-RobotManip represents states and actions uniformly as 80-dimensional vectors. The first 58 dimensions are divided equally between two arms, each with seven joint-position dimensions, nine end-effector-pose dimensions, one gripper-state dimension, and twelve dexterous-hand-joint dimensions; the remaining 22 are reserved. Missing components are padded with zeros.

Because padded zeros are not real data, a binary 0–1 mask identifies padded dimensions. Only dimensions that actually exist contribute to errors and gradient updates during training.

#### 2. Coordinate-System Alignment

Suppose two robots both move an arm ten centimeters forward. Different base mounting positions, orientations, and coordinate conventions may encode that motion as entirely different numbers in their respective base frames. Qwen-RobotManip therefore favors changes in end-effector pose in camera coordinates over absolute positions in original coordinates, avoiding problems from different origins. The model also receives camera mounting positions and orientations and end-effector types.

#### 3. Providing Current-Robot Characteristics in Context

The preceding discussion addresses learning effectively from different robots. At inference, however, the model faces one specific robot, and we want strong specialization: it should infer what robot it is handling from a short execution history. Besides images, it therefore receives robot platform, operating speed, FPS, and recent observations and actions, allowing temporary adaptation to the current embodiment through an in-context-learning-like method.

### (2) Aligning Modalities within Each Data Sample

Even after all robots use a unified representation, each sample's modalities must be mutually consistent. A robot-training sample usually contains images, other states, actions, and text instructions. Real data often has misalignment: video and robot logs are not fully synchronized, some action chunks deviate substantially from reality, or instructions contradict images and actions.

Qwen-RobotManip does not attempt to check all four modalities for complete consistency at once. Instead, it ensures pairwise alignment: one pipeline checks state–action consistency, another checks instruction–image consistency, and another checks image–state consistency. Each check handles only part of the relationships, but repeated filtering leaves substantially more coherent data overall.

### (3) Data Cleaning and Correctness Checks

Qwen-RobotManip checks robot logs themselves, first verifying plausible temporal changes in states and actions. If a robot's state consistently moves in one direction while actions show the opposite trend, the data may be wrong. A joint angle suddenly jumping from a normal value to an extreme may likewise indicate sensor or recording errors. After filtering these anomalies, the robot's kinematic model checks whether end-effector positions computed from joint states match recorded data. Coordinate directions across datasets are also unified.

These steps are necessary. Without prior cleaning, many erroneous samples enter training, and the model learns collection errors instead of how the robot should actually act.

## References

- 具身纪元. (2026). [The Secret of Pretraining Lies in More Than Scale and Diversity](https://www.jintiankansha.com/t/8QXVctCOWG) (translated title; in Chinese). WeChat public-account article, published July 4, 2026.
- Physical Intelligence. (2026). $\pi_{0.7}$: [A Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://arxiv.org/abs/2604.15483). arXiv:2604.15483.
- Yuan, H., Liang, Z., Chen, A., et al. (2026). [Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models](https://arxiv.org/abs/2606.17846). arXiv:2606.17846.
