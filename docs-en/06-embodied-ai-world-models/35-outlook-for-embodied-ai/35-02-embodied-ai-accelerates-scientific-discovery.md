---
title: "35.2 Embodied AI Accelerates Scientific Discovery"
chapter_title: "Outlook for Embodied AI"
section_id: "35-02"
language: en
source_language: zh
source_docx: "第6部分 具身智能与世界模型/35.具身智能的展望/35.2 具身智能加速科学发现.docx"
status: "manually rebuilt and checked against Word"
ocr: "all Word-visible text and formula images manually transcribed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 35.2 Embodied AI Accelerates Scientific Discovery

## I. Scientific Discovery with Physical APIs

In natural science, medicine, and related fields, the core bottlenecks for AI for Science are data and experimental validation. Once many promising methods and ideas can be proposed, validation becomes the bottleneck. Even if simulations reduce the need for wet experiments, building those simulations is constrained not by models but by large quantities of high-quality, reproducible perturbation data. Only embodied AI can autonomously validate experiments and obtain data.

1. Closing the dry–wet loop of exploratory research, freeing human researchers, and reducing exploration costs

Today's automated laboratories with hard-coded robots cover narrow scenarios, take a long time to customize, and are expensive. They therefore often appear in the least novel directions and struggle to accelerate discovery in long-tail areas. Embodied AI can execute experiments flexibly like a human, adjusting actions in real time to perceived sample states and accumulating tacit knowledge embedded in “feel” rather than literature, such as forming a water film of just the right thickness. Sequencing and imaging for cellular multi-omics are already automated, but embodied AI can also handle preprocessing requiring real-time observation-based judgment, such as tissue dissociation, single-cell suspension preparation, primary-cell handling, and iPSC/organoid culture. Core gene-editing steps are automated, but embodied AI can additionally connect downstream clonal expansion and phenotype confirmation. It will also perform soft-tissue sectioning, animal experiments, transfers between instruments, new experimental procedures, and troubleshooting. Once deployed, embodied AI will take the AI Scientist toward a “general scientist loop,” analogous to moving from specialized to general-purpose processors. Exploratory research will no longer depend so heavily on human labor, freeing researchers for creative thought. Generality also greatly reduces per-exploration costs and enables closed-loop optimization, potentially yielding more discoveries.

2. Accelerating nonstandard experiments

Throughput has already grown by orders of magnitude for many experiments. Yet Amdahl's law means that the slowest nonstandard experiments remain decisive. Humans often work efficiently for only a few hours per day and have high experimental failure rates. Embodied AI can operate 7\*24 hours and continually reduce failure rates, accelerating individual experiments.

3. Bypassing human-attention bottlenecks to expand parallelism

Computer science teaches that parallelism improves efficiency when individual tasks take a long time. Embodied AI can parallelize nonstandard experiments at scale without stopping for physical events designers did not enumerate, such as clogged nozzles, precipitation, contamination, or abnormal cell states. If humans must handle all of these, achievable parallelism is limited by their serial efficiency. Apart from a few experiments requiring serial execution because of causal dependencies or resource constraints, most can run massively in parallel in the physical world. Some model training also requires data acquired this way, such as diverse perturbation data for virtual-cell models, which decentralized human data production cannot provide.

4. Improving data quality to simplify reproduction, enable simulation, and increase model accuracy

Specialized models, and even simulations, depend on large-scale, high-quality data, especially causal perturbation data. Yet experiments often vary markedly across operators. In plate streaking, for example, differences in the number and manner of strokes directly affect colony density, producing inconsistent data from which good models are difficult to train. Embodied AI can make manipulation of soft tissues, animals, microorganisms, and other subjects traceable and standardized, making wet experiments reproducible with marginal reproduction costs approaching zero. It can reduce batch effects and collect negative samples often missing or ignored in human research, providing high-quality data for specialized models. This compounds: better data yields better models, better models predict more accurately, and more accurate results support collection of higher-quality data, greatly accelerating research.

5. Capturing scientific discoveries in “accidents”

Many natural-science discoveries arise from accidents. Laboratories relying on hard-coded robots ignore unexpected events or stop and wait for humans, while transient phenomena may disappear quickly. Embodied AI can actively perceive and acquire relevant data, such as observing from another angle, potentially enabling new discoveries.

## II. Scientific Discovery through Physical Self-Improvement

Self-improving embodied AI marks a transition toward inventing tools like humans. It can not only conduct experiments but modify its experimental interfaces and create new measurement methods and experimental techniques, continually expanding its ability to obtain real-world information and improving how it conducts scientific research.

1. Improving experimental operations and equipment

Self-improving embodied AI can autonomously refine operations: observing minute morphological changes in real time, combining mechanical feedback to precisely control the speed of medium changes or gripping force at every moment, or even performing local microsurgery. Adaptation through methods such as model fine-tuning can substantially reduce failed experiments and repetitions. It can also improve equipment, design more efficient devices, and increase throughput and accuracy.

2. Accelerating idea validation and engineering iteration when exploring new experimental methods and measurement tools

Breakthroughs in experimental methods and measurement tools often yield important discoveries because they bypass intrinsic complexity in the subject and reveal the essence of a problem directly. Future models may propose many ideas for new methods and tools, some workable and some not, but validation and engineering improvement become bottlenecks that prevent a closed loop. Embodied AI can explore and iteratively improve in parallel over a broader space at speeds far exceeding humans, handle procedures beyond hard-coded robots, optimize valuable methods, and quickly eliminate ineffective ones. This greatly shortens the path from idea to use in real experimental environments while improving performance.

3. Designing and manufacturing corresponding narrow-domain automation after discovering new experimental pathways

Embodied AI and narrow-domain automation are not independent paths. Once general embodied AI can self-improve, it can design, manufacture, and improve specialized automation. In future laboratories, embodied AI will therefore not always conduct experiments itself. It will explore, then “compile” stable workflows into high-throughput automated infrastructure for subsequent experiments, continuing other exploration while monitoring as appropriate.

4. Using intuition accumulated through physical interaction to invent technologies and even propose theories

Embodied AI understands physical-world dynamics and spatial structures and can accumulate tacit knowledge or intuition through many experiments. In experimental science, it understands practical difficulties and unwritten knowledge, resolving communication problems between experimental scientists and engineers. Creative scientific intuition about “what is worth doing” may also relate to embodied experience and tacit knowledge acquired through immersion in laboratories. Even in theoretical physics, humans abstracted conceptual frameworks such as force and causality from raw data through repeated direct interaction with the physical world. LLMs cannot, like Newton, abstract the complex interactions of everything into a unified “force”: their understanding comes from pretraining data, and they can only interpolate and combine concepts and frameworks already present in those data. This does not yield entirely new concepts or theories, and reinforcement learning does not confer conceptual-level innovation either. Embodied AI can actively interact with reality and autonomously obtain information beyond existing frameworks, potentially a necessary condition for conceptual innovation.

## References

- Szymanski, N. J., Rendy, B., Fei, Y., et al. (2023). [An Autonomous Laboratory for the Accelerated Synthesis of Inorganic Materials](https://www.nature.com/articles/s41586-023-06734-w). Nature, 624, 86–91.
- Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292). arXiv:2408.06292.
- Canty, R. B., & Abolhasani, M. (2026). [The Past, Present and Future of Self-Driving Laboratories](https://www.nature.com/articles/s41570-026-00847-2). Nature Reviews Chemistry, 10, 523–537.
- Fan, J. (2026). [Robotics' End Game: Nvidia's Jim Fan](https://www.youtube.com/watch?v=3Y8aq_ofEVs). Sequoia Capital AI Ascent 2026 [Video].
