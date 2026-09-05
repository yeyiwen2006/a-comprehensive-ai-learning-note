---
title: "5.7 Zeroth-Order Optimization Algorithms"
chapter_title: "Optimization Algorithms"
section_id: "05-07"
language: en
source_language: zh
source_docx: "第1部分 深度学习/5.优化算法/5.7 零阶优化算法.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 5.7 Zeroth-Order Optimization Algorithms

When gradients (first-derivative information) cannot be obtained, zeroth-order optimization can be used.

## I. Genetic Algorithms

A genetic algorithm is a search heuristic inspired by Darwinian evolution and "natural selection and survival of the fittest." It simulates processes from natural selection and genetics to solve optimization and search problems. Its core idea is simple: within a "population" of candidate solutions ("individuals" or "chromosomes"), fitter (better) individuals have higher chances of surviving, reproducing, and passing their "genes" (solution features) to the next generation. Across generations, the population's overall fitness improves, eventually converging to one or several excellent solutions.

1. Basic concepts and terminology

Individual/chromosome: a possible solution, generally encoded as a string, such as a binary string or a sequence of floating-point numbers.

Gene: an element of a chromosome representing a solution feature or parameter.

Population: a collection of individuals processed in each generation.

Fitness function: an evaluation function measuring solution quality. Higher fitness indicates a better individual.

Selection: choose good individuals from the current population as "parents" for the next generation according to fitness. Fitter individuals are more likely to be selected.

Crossover: simulate sexual reproduction. Two parent chromosomes exchange some genes to produce one or two new offspring. This is the main way to generate individuals.

Mutation: randomly change one or more genes with a small probability. This introduces new genetic material, maintains population diversity, and helps avoid premature convergence to local optima.

2. Algorithm workflow (a five-step loop)

Genetic algorithms generally follow this iterative procedure:

Detailed steps:

Initialization: randomly generate an initial population of N individuals, like a first generation of organisms with random genes.

Fitness evaluation: evaluate every individual using the fitness function. For function maximization, the function value itself can serve as fitness.

Selection: choose "excellent" parents from the current population, often through roulette-wheel selection. Imagine a wheel on which each individual's area is proportional to its fitness; fitter individuals have a greater selection probability.

Crossover: combine parental features to produce offspring, hoping to combine their advantages. Single-point crossover is most common: choose a point randomly, cut both chromosomes there, and exchange their latter halves. (Parent 1: 1010 | 1010; parent 2: 1100 | 1100; offspring 1: 1010 1100; offspring 2: 1100 1010.)

Mutation: introduce random changes to increase diversity and explore new solutions. With a small probability (such as 0.1%), randomly alter an offspring gene. With binary encoding, flip 0 to 1 or 1 to 0.

Replace the old population with the new offspring, forming the next generation, and return to step 2 (fitness evaluation). Repeat until a stopping condition is met: a maximum number of iterations (generations), no significant improvement in the best individual's fitness over several generations, or a satisfactory solution (such as fitness exceeding a threshold).

3. A simple example: maximizing f(x) = x² on [0, 31]

Encoding: represent x with five binary digits. For example, 11001 represents 25 and 00000 represents 0.

Initialization: randomly generate four chromosomes.

A: 01101 (13) B: 11000 (24) C: 01000 (8) D: 10011 (19)

Fitness evaluation: f(x) = x²; f(A) = 169, f(B) = 576, f(C) = 64, f(D) = 361.

Selection: select parents proportionally to fitness. B and D are highly fit and more likely to be selected. Suppose B and D form one pair and A and D another.

Crossover: suppose the cut is after the third digit.

Parents (B, D): 110 | 00 and 100 | 11.

Offspring (E, F): 11011 (27) and 10000 (16).

Parents (A, D): 011 | 01 and 100 | 11.

Offspring (G, H): 01111 (15) and 10001 (17).

Mutation: suppose the last bit of offspring F (10000) mutates to produce 10001 (17).

The new population is E(27), F(17), G(15), H(17). Their fitness values are:

f(E) = 729, f(F) = 289, f(G) = 225, f(H) = 289.

After just one generation, average and best fitness have both improved substantially. Continued iteration eventually converges to 11111 (31), the maximum.

4. Advantages, disadvantages, and applications

Advantages:

Strong global search: less likely to become trapped in local optima, particularly suitable for complex nonlinear problems.

Generality: few requirements on mathematical properties such as differentiability or continuity; an encoding and fitness function suffice.

Implicit parallelism: process multiple population points simultaneously for efficient search.

Robustness: insensitive to the initial solution.

Disadvantages:

Slow convergence: for simple problems, it may be slower than conventional methods such as gradient descent.

Complex parameter tuning: crossover rate, mutation rate, and population size strongly affect results and require experience to tune.

"Guaranteeing optimality": the global optimum is not guaranteed, only a "satisfactory" approximation.

Encoding and fitness-function design are critical; poor design degrades performance.

Applications:

Function and combinatorial optimization: traveling-salesperson problems and scheduling (vehicle and job-shop scheduling).

Artificial intelligence: neural architecture design and fuzzy-rule optimization.

Automatic programming: design computer programs to satisfy specified requirements.

Machine learning: feature selection and classifier-parameter tuning.

Economics and finance: prediction models and portfolio optimization.

Bioinformatics: DNA-sequence analysis and protein-structure prediction.

Engineering: aircraft-wing design, antenna design, and robot path planning.

Summary: genetic algorithms are powerful, flexible global-optimization tools that simulate natural evolution to efficiently search enormous spaces for "good enough" solutions. They may not find the absolute optimum or be the most computationally efficient, but offer distinct advantages for complex problems lacking ready-made mathematical models or resisting conventional methods.

## References

- Holland, J. H. (1975). [Adaptation in Natural and Artificial Systems](https://openlibrary.org/books/OL5070129M/Adaptation_in_natural_and_artificial_systems). University of Michigan Press.
- Goldberg, D. E. (1989). [Genetic Algorithms in Search, Optimization, and Machine Learning](https://openlibrary.org/books/OL2030750M/Genetic_algorithms_in_search_optimization_and_machine_learning). Addison-Wesley.
- Nesterov, Y., & Spokoiny, V. (2017). [Random Gradient-Free Minimization of Convex Functions](https://doi.org/10.1007/s10208-015-9296-2). Foundations of Computational Mathematics.
