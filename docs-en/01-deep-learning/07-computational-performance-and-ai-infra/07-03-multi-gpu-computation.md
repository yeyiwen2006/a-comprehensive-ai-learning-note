---
title: "7.3 Multi-GPU Computation"
chapter_title: "Computational Performance and AI Infra"
section_id: "07-03"
language: en
source_language: zh
source_docx: "第1部分 深度学习/7.计算性能与AI Infra/7.3 多GPU计算.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 7.3 Multi-GPU Computation

## I. Three Partitioning Strategies

1. Network partitioning (partitioning layers/network depth, vertical partitioning): pipeline parallelism (PP)

(1) Principle: assign different neural network layers to different GPUs. For example, GPU 1 handles layers 1–10 and GPU 2 handles layers 11–20. Data flow through the GPUs sequentially, like a pipeline.

(2) Advantages

Supports extremely large models: different parts of the model reside on different GPUs, so no single GPU needs to store the entire model, enabling models with enormous parameter counts to be trained.

(3) Disadvantages

Load imbalance: layers have strong dependencies, so a layer must finish computing before the next can proceed. This readily leaves GPUs idle (waiting), and it is difficult to ensure that each layer's workload keeps every GPU busy and synchronized.

Transfer bottleneck: the volume of data (activations and gradients) transferred between layers is enormous and may exceed inter-GPU bandwidth limits.

(4) Conclusion: generally not recommended; used only in very large model pretraining (together with other methods), not in forward inference (because at the same compute capacity, this method is slowest, increasing latency and harming the user experience).

2. Within-layer partitioning (partitioning channels/width, horizontal partitioning): tensor parallelism (TP)

(1) Principle: split the computational work within a layer. For matrix multiplication Y = W*X, for example, divide the weight matrix W into W_1 and W_2. GPU 1 holds W_1 and computes part of the result; GPU 2 holds W_2 and computes the other part. They then exchange data instantly over an extremely fast interconnect (NVLink) and add the results to obtain the complete token. As another example, a convolutional layer with k channels can be assigned to k GPUs, each responsible for computing a subset of channels. Within-layer partitioning is not limited to weights: K and V in the KV cache are also partitioned this way during long inference.

(2) Advantages

Linear scaling of device memory: memory usage is distributed, so increasing the number of GPUs supports wider networks with larger parameter counts.

Performance improvement: computational performance improves substantially when the weights are large.

(3) Disadvantages

Extremely high synchronization cost: each layer often involves feature reorganization and interaction (channel reconstruction in convolution, for example), so all GPUs' results must be aggregated after each layer before proceeding to the next (a barrier operation), causing extensive waiting.

High communication overhead: the amount of transferred data may exceed that of the first method.

(4) Conclusion: generally used only for large model pretraining and inference, when a single GPU cannot perform the computation or hold all weights in device memory, making partitioning unavoidable. For small models, this method has poor cost-effectiveness because of bandwidth costs and synchronization complexity.

3. Data parallelism (DP; partitioning data)

(1) Principle

Model replication: every GPU retains a complete copy of the model parameters.

Data partitioning: divide a large batch into smaller portions and distribute them to different GPUs. For example, to complete masked prediction for 100 articles, each GPU handles 10 articles (note: within each GPU, the first word of every article is also generated in parallel, then the second word of every article, and so on). In RLHF, the system draws 128 questions from the prompt pool; GPU 1 receives questions 1–32 and GPU 2 receives questions 33–64. Although GPU 1 must generate the first character, second character, and so on sequentially, within the same millisecond it computes the T-th character of all 32 sentences simultaneously, as does GPU 2.

Independent computation, unified updates: each GPU independently computes gradients for its own data. Gradients from all GPUs are then aggregated (summed or averaged), and the aggregated gradients update the model parameters on every GPU. In the RLHF example above, GPUs need to stop and exchange information only after everyone has finished writing the sentences and is about to compute gradients to update the model.

(2) Advantages

Simple implementation: easiest to put into practice and applicable to almost every situation.

High efficiency: synchronization occurs only after processing a minibatch, and differences in gradient-computation timing can hide part of the communication.

High throughput: more GPUs allow more data to be processed at once (a larger batch size), accelerating training.

(3) Disadvantages

Limited by a single GPU's memory: since each GPU must store the complete model, this method cannot train extremely large models that do not fit into one GPU's memory.

(4) Conclusion: the most commonly used method. With increasing memory capacity in modern GPUs, it is now used in training and inference for almost all models.

(5) Workflow

Distribute data: split a random minibatch evenly into k portions.

Local computation: each GPU computes the loss and gradients using its assigned data.

Gradient aggregation: collect and combine the gradients computed by the k GPUs.

Gradient broadcast: send the combined total gradient back to every GPU.

Parameter update: every GPU updates its own model parameters using the same total gradient, ensuring that all GPU models remain consistent.

(6) Key adjustments in practice

Batch size: when training on k GPUs, each GPU has the same workload as in single-GPU training, so the total batch size is typically increased by a factor of k.

Learning rate: the larger batch size yields a more accurate gradient estimate, so the learning rate usually needs to increase accordingly to accelerate convergence.

Batch normalization: this is a special case. BN depends on batch statistics (mean and variance), which may be inaccurate when computed directly on one GPU because its local data volume is small. Adjustments are usually needed, such as synchronized BN (SyncBN) or separate BN parameters for each GPU.

## II. Zero Redundancy Optimizer (ZeRO)

This is another technology currently used in large model pretraining (at the core of DeepSpeed/FSDP).

Standard data parallelism stores a complete model on every GPU, wasting too much device memory. ZeRO “slices up” model parameters so that each GPU stores a small portion. Unlike within-layer partitioning, however, in TP each GPU computes “input * this GPU's parameters,” and the results are then added. In ZeRO, GPU 1 first borrows all needed parameters from other GPUs, performs “input * complete parameters,” and immediately discards any parameters it does not own after computing (saving device memory).

TP partitions not only weight **storage**, but, more importantly, the **computational process**. It changes how matrix multiplication is mathematically executed.

- **Scenario**: we need to compute $Y = X \times W$, but matrix $W$ is too large.
- **Method**:
  - Split $W$ vertically into $W_A$ and $W_B$.
  - GPU 1 holds $W_A$ and computes $Y_A = X \times W_A$.
  - GPU 2 holds $W_B$ and computes $Y_B = X \times W_B$.
- **Key points**:
  - **Needed at every moment**: throughout this layer's computation, both GPU 1 and GPU 2 are working.
  - **Extremely frequent communication**: immediately after computing this layer, an `All-Reduce` (aggregation) over NVLink is required to obtain the final $Y$ for the next layer.
  - **Purpose**: accelerate a single matrix computation or handle a computation that one GPU cannot perform.

ZeRO is fundamentally still **data parallelism**. This means that, logically, every GPU should have a **complete** model copy. ZeRO says: “That is too wasteful. Let us pretend everyone has a complete copy while actually storing only part of it each.”

- **Scenario**: we are using data parallelism. GPU 1 handles data A, and GPU 2 handles data B.
- **Method (ZeRO-3 as an example)**:
  - Parameters $W$ are sharded: GPU 1 stores the first half and GPU 2 the second. **Note: this is the state during static storage.**
  - **When computation starts (forward propagation)**:
    1. **Broadcast (borrowing parameters)**: on reaching layer 1, GPU 1 finds it has only half the parameters and immediately asks GPU 2 for the other half. GPU 2 does likewise.
    2. **Reassembly**: both GPU 1 and GPU 2 now have the **complete parameters of layer 1** in device memory.
    3. **Independent computation**: GPU 1 processes its data A, and GPU 2 processes its data B. **Note: they are not jointly computing one matrix; each processes its own data.**
    4. **Discarding**: after computing layer 1, everyone immediately deletes the borrowed parameters to save memory, retaining only the small shard it is responsible for storing.
- **Key points**:
  - **Temporary reassembly**: parameters are complete during computation.
  - **Communication for device memory**: bandwidth (repeatedly borrowing and discarding parameters) is exchanged for memory capacity.

In one sentence: TP divides the task of “doing the work” (everyone computes together); ZeRO divides the task of “keeping the materials” (everyone takes turns using them).

## III. Combining Multiple Methods

When training large models, we usually combine multiple methods:

1. Use data parallelism to partition data by batch;

2. Partition the model within layers (for example, splitting one model across eight GPUs) to make matrix multiplication feasible and keep single-layer computation latency low;

3. Then use ZeRO across different machine nodes. Without ZeRO, each TP group must store complete optimizer states during training (FP32 parameter backups, momentum, and variance), which occupy several times as much device memory as the model parameters alone. With ZeRO, these states can be sharded across hundreds or thousands of GPUs.

4. Use between-layer partitioning when necessary.

## References

- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). [Dive into Deep Learning](https://D2L.ai). Cambridge University Press.
- Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054). SC 2020.
