---
title: "27.1 Flow-Matching Models"
chapter_title: "Flow-Matching Models"
section_id: "27-01"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/27.流匹配模型/27.1 流匹配模型.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 27.1 Flow-Matching Models

## I. The Basic Idea of Flow Matching and Its Comparison with Diffusion Models

Flow matching and diffusion models are both generative models. Both aim to transform a simple distribution (such as Gaussian noise, $p_0$) into a complex data distribution (such as images or text, $p_1$).

The idea of a diffusion model is that data gradually becomes noise as noise is added, much like ink diffusing in water. Generation reverses this diffusion process, recovering the data step by step through “denoising.” This is usually modeled as a stochastic differential equation (SDE), with a discrete, winding path.

The idea of flow matching is to obtain the distribution before noise addition directly from the noisy distribution. It directly defines a vector field that tells particles (data points) which direction to move at each time $t$.

Several basic concepts in flow matching:

- **Initial state**: at $t=0$, all samples follow a simple distribution $p_0(x)$, such as standard Gaussian noise, like a loose pile of sand.
- **Target state**: at $t=1$, the samples come together to form the complex data distribution $p_1(x)$ we want to generate, such as a collection of beautiful images, like sand forming a castle.
- **Velocity field**: denoted by $v(t,x)$, this is a “weather vane” or “force field” at any time $t\in[0,1]$ and position $x$ in high-dimensional space. It indicates the direction and speed at which a particle at that position should move in the next instant.
- **The role of the ordinary differential equation**: the rate of change of a particle's trajectory $x_t$ over time (its velocity) is determined by the velocity field at its location. This is naturally expressed mathematically as an ordinary differential equation:

$$
\frac{dx_t}{dt}=v(t,x_t)
$$

If we know the wind direction at every time and position (the velocity field learned by the neural network), we must integrate it (solve this ODE) to determine where the grain of sand will eventually be blown.

We want a particular data point $a$ to move from $x_0$ (a known noise position) toward $x_1$ (an unknown meaningful position), following a given velocity distribution (the value of the “velocity field” at the corresponding location and time). In fact, the ODE above can be written in integral form:

$$
x_1-x_0=\int_0^1 v_t\,dt
$$

Because we do not care what happened during noise addition, we can assume an ideal case in which all data points move in straight lines at constant speeds in the desired directions. Our goal is therefore to learn a velocity field that approximates this ideal. If we place many noise points into the field, they will eventually converge into meaningful data.

## II. Mathematical Principles

The core of flow matching is to define and learn this “velocity field.”

### A. Probability Path

We do not need to simulate a complex physical diffusion process as in diffusion models. Instead, we can directly define a path from noise $x_0$ to data $x_1$. The simplest and most efficient approach is linear interpolation:

$$
x_t=(1-t)x_0+tx_1,\quad t\in[0,1]
$$

- At $t=0$, $x_t=x_0$, entirely noise.
- At $t=1$, $x_t=x_1$, entirely data. This path is straight, which is important for fast generation.

### B. Velocity Field

Once we have a formula for position $x_t$ as a function of time $t$, we can calculate the particle's velocity (by differentiating with respect to time):

$$
u_t(x_t\mid x_0,x_1)=\frac{d}{dt}x_t=x_1-x_0
$$

This $u_t$ is the target velocity in conditional flow matching. It tells us the speed and direction in which the current particle should move to reach $x_1$ at $t=1$.

### C. Flow-Matching Objective

During training, we cannot know the endpoint $x_1$ in advance; it is what we want to generate. We therefore need to train a neural network $v_\theta(x_t,t)$ to predict the target velocity given only the current position $x_t$ and time $t$.

The loss is very simple: it is a regression problem:

$$
\mathcal{L}_{FM}(\theta)=\mathbb{E}_{t,x_0,x_1}\left[\left\|v_\theta(x_t,t)-u_t(x_t\mid x_0,x_1)\right\|^2\right]
$$

In other words, the “flow velocity” predicted by the neural network should be as close as possible to the true “flow velocity.”

In practice, of course, for tasks such as image generation, the information available when predicting $v$ includes the current position $x_t$, time $t$, and condition $c$.

Given an endpoint, we want the velocity to be constant and directed toward it. Why, then, do we still need a neural network to fit it? Because at inference time we do not know the endpoint. Different diffusion starting points (corresponding to different “particles”) and different additional conditions lead to different endpoints and different “ideal velocities.” In a vast high-dimensional vector field, however, the actual velocity for a particular task (the field direction at that position) is often not constant and pointed straight at the endpoint. A flow-matching model attempts to fit the desired velocity direction at each given point as closely as possible.

## III. Training Method (Using a Flow-Matching VLA Model as an Example)

Consider current state-of-the-art optimal transport flow matching (similar to the design in $\pi_0$). Suppose the base noise distribution is $\pi_0\sim p_0=\mathcal{N}(0,I)$, the real action-trajectory distribution is $\pi_1\sim p_1$, and the multimodal condition is $c=\mathrm{Encoder}(o_t,I_t)$.

The model constructs a straight probability path connecting noise to real actions:

$$
x_t=(1-t)x_0+tx_1
$$

The action generator $v_\theta$ aims to fit the true velocity field, with objective:

$$
\mathcal{L}_{FM}(\theta)=\mathbb{E}_{x_0,x_1,t}\left[\left\|v_\theta(x_t,t,c)-(x_1-x_0)\right\|^2\right]
$$

Here, $(x_1-x_0)$ is the constant velocity of the true path. Compared with traditional DDPM diffusion models, the ODE trajectories of flow matching are smoother and straighter, enabling high-fidelity continuous action signals to be generated in fewer inference steps (even just a few).

During training, for each pair $x_0$ and $x_1$, we want the velocity at every point on their connecting line to point from $x_0$ to $x_1$, with a magnitude numerically equal to their difference. In practice, because the velocity field in space is continuous, these requirements can only be approximated. When multiple connecting lines converge at one point, its velocity is approximately the average of the velocities associated with the lines.

Workflow:

1. **Feature extraction**: the VLM receives the current camera observation $o_t$ and language $l$, then performs a forward pass to generate the multimodal conditioning vector $c$.
2. **Noise initialization**: sample Gaussian trajectory noise $x_0\sim\mathcal{N}(0,I)$ of length $H$ from a standard normal distribution.
3. **ODE solving**: over time steps $t\in[0,1]$, the action generator $v_\theta(x_t,t,c)$ uses context $c$ to compute the gradient of the current state. A numerical integrator (such as Euler's or Heun's method) updates $x_t$ step by step.
4. **Action execution**: take the ODE endpoint $x_1$ as the predicted action chunk $A_t$. The system uses model predictive control (MPC / receding-horizon control), executing only the first $k$ actions before obtaining a new visual observation and starting the next round of closed-loop inference.

Time is continuous here, but the neural network can only process discrete $t$ inputs. Sampling is therefore often used: for example, a sample is taken every $0.02$ time units, and the network output at that time is integrated over the interval to move $x_t$ “one small step.”

## IV. Applications

1. Generating high-resolution, high-quality images, with performance comparable to or better than diffusion models such as Stable Diffusion and usually faster generation (because straighter trajectories require fewer sampling steps).

2. Generating continuous sequences of video frames and handling high-dimensional spatiotemporal data.

3. Scientific computing and simulation, such as simulating physical systems (for example, fluid dynamics) and generating molecular conformations (protein 3D structure prediction). These problems essentially involve modeling changes in continuous variables (how the velocity of points in high-dimensional space varies over space and time).

## References

- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). ICLR.
- Liu, X., Gong, C., & Liu, Q. (2023). [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003). ICLR.
