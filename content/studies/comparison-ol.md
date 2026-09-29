---
title: "Bandit Algorithms: Methods & Implementation Review"
date: 2023-10-20
lastmod: 2026-09-29
category: "Online Learning"
tags: ["Bandits", "UCB", "Thompson Sampling"]
author: ["Yong-Hwan Lee"]
summary: "Compare exploration strategies and inspect where the experimental implementations differ from standard UCB, Thompson sampling, and nonlinear bandit methods."
editPost:
    URL: "https://github.com/kapshaul/OnlineLearning/blob/main/docs/bandits-comparison.md"
    Text: "GitHub"
---

## Overview

This Oregon State University study explores how bandit algorithms trade off collecting information and choosing actions with high estimated rewards. It covers independent-arm methods, linear contextual methods, and a squared-reward experiment.

The tables and figures preserve the original recorded results. A source review found differences between several implementations and their standard algorithm names. Those differences are explained alongside the results so that the numbers are interpreted as observations of the implemented variants.

## Experiment and metric

For expected reward $\mu_t(i)$ and selected arm $A_t$, cumulative pseudo-regret is

$$
R_T=\sum_{t=1}^{T}\left[\max_{i\in\mathcal A_t}\mu_t(i)-\mu_t(A_t)\right].
$$

The simulator adds the same sampled noise to the selected reward and the comparator reward, so that noise cancels in the recorded difference. With multiple users, its plotted cumulative regret is averaged across users.

The linked simulation's defaults are 25 context dimensions, 25 arms, 10 users, 200,000 iterations, and Gaussian observation noise with standard deviation 0.1. The archive does not pin a seed and complete configuration for every figure, so these defaults should not be assumed to describe every historical run.

A further logging detail matters: the source records every 100 iterations starting at zero and prints the last recorded point. That point need not equal the full-horizon total. The tables below retain the **reported cumulative regret**, rather than claiming a newly verified final-horizon result.

## Explore-then-commit and UCB

Explore-then-commit first gathers observations, then chooses the best empirical arm for the remaining rounds. Its exploration parameter $m$ controls the initial sampling budget. Too little exploration risks a poor commitment; more exploration also incurs a cost.

| Explore-then-commit setting | Reported regret |
| --- | ---: |
| $m=10$ | 1001.40 |
| $m=20$ | 214.90 |
| $m=30$ | 334.02 |

<details>
<summary>View explore-then-commit curves</summary>
<div class="study-figure-grid">
<figure><img src="/online-learning/img/ETC10.png" alt="Explore-then-commit regret, m = 10"><figcaption>Explore-then-commit regret, m = 10</figcaption></figure>
<figure><img src="/online-learning/img/ETC20.png" alt="Explore-then-commit regret, m = 20"><figcaption>Explore-then-commit regret, m = 20</figcaption></figure>
<figure><img src="/online-learning/img/ETC30.png" alt="Explore-then-commit regret, m = 30"><figcaption>Explore-then-commit regret, m = 30</figcaption></figure>
</div>
</details>

UCB keeps exploring through a confidence bonus. For an arm with $N_i>0$ previous observations, the implementation uses

$$
\operatorname{score}_t(i)=\widehat\mu_i+
\sqrt{\frac{2\alpha\log t}{N_i}},
$$

where $t$ is the number of completed updates. Unobserved arms are selected before this expression is evaluated. The factor $\alpha$ was missing from the previous explanation, even though it changes the experiment.

| UCB setting | Reported regret |
| --- | ---: |
| $\alpha=0.1$ | 256.50 |
| $\alpha=0.5$ | 977.03 |
| $\alpha=1.0$ | 1906.65 |

<details>
<summary>View UCB curves</summary>
<div class="study-figure-grid">
<figure><img src="/online-learning/img/UCB01.png" alt="UCB regret, alpha = 0.1"><figcaption>UCB regret, alpha = 0.1</figcaption></figure>
<figure><img src="/online-learning/img/UCB05.png" alt="UCB regret, alpha = 0.5"><figcaption>UCB regret, alpha = 0.5</figcaption></figure>
<figure><img src="/online-learning/img/UCB1.png" alt="UCB regret, alpha = 1.0"><figcaption>UCB regret, alpha = 1.0</figcaption></figure>
</div>
</details>

These runs show sensitivity to exploration settings. They do not establish that the smallest exploration bonus is generally best.

## Gaussian reward sampling

The class named `ThompsonSamplingGaussianMAB` draws an independent score for each arm using

```python
np.random.normal(empirical_mean, 1 / (observation_count + 1))
```

NumPy's second argument is the **standard deviation**, not the variance. Therefore, the implemented score distribution is

$$
\widetilde\mu_i\sim
\mathcal N\left(\widehat\mu_i,\frac{1}{(N_i+1)^2}\right),
$$

when the second parameter in $\mathcal N$ denotes variance.

The center is an empirical mean, and the shrinking scale is chosen directly in code. Without a corresponding prior and observation model, this is a Gaussian sampling heuristic rather than a derived Bayesian posterior update.

| Archived method label | Reported regret |
| --- | ---: |
| Thompson Sampling | 100.00 |

<details>
<summary>View the Gaussian sampling curve</summary>
<div class="study-figure-grid">
<figure><img src="/online-learning/img/TS.png" alt="Gaussian sampling heuristic: cumulative regret"><figcaption>Gaussian sampling heuristic: cumulative regret</figcaption></figure>
</div>
</details>

## Linear contextual methods

With context vectors $x_s$, observed rewards $r_s$, and regularization $\lambda>0$, define

$$
A_t=\lambda I+\sum_{s<t}x_sx_s^\top,\qquad
b_t=\sum_{s<t}x_sr_s,\qquad
\widehat\theta_t=A_t^{-1}b_t.
$$

This is a regularized least-squares estimator for a linear reward model.

### LinUCB

LinUCB evaluates an available context $x$ using

$$
x^\top\widehat\theta_t+\alpha\sqrt{x^\top A_t^{-1}x}.
$$

| LinUCB setting | Reported regret |
| --- | ---: |
| $\alpha=0.5$ | 24.43 |
| $\alpha=1.5$ | 177.89 |
| $\alpha=2.5$ | 487.73 |

<details>
<summary>View LinUCB regret and parameter error</summary>
<div class="study-figure-grid">
<figure><img src="/online-learning/img/LinUCB05.png" alt="LinUCB regret, alpha = 0.5"><figcaption>LinUCB regret, alpha = 0.5</figcaption></figure>
<figure><img src="/online-learning/img/LinUCB15.png" alt="LinUCB regret, alpha = 1.5"><figcaption>LinUCB regret, alpha = 1.5</figcaption></figure>
<figure><img src="/online-learning/img/LinUCB25.png" alt="LinUCB regret, alpha = 2.5"><figcaption>LinUCB regret, alpha = 2.5</figcaption></figure>
<figure><img src="/online-learning/img/LinUCB05_est.png" alt="LinUCB parameter error, alpha = 0.5"><figcaption>LinUCB parameter error, alpha = 0.5</figcaption></figure>
<figure><img src="/online-learning/img/LinUCB15_est.png" alt="LinUCB parameter error, alpha = 1.5"><figcaption>LinUCB parameter error, alpha = 1.5</figcaption></figure>
<figure><img src="/online-learning/img/LinUCB25_est.png" alt="LinUCB parameter error, alpha = 2.5"><figcaption>LinUCB parameter error, alpha = 2.5</figcaption></figure>
</div>
</details>

### Linear Thompson sampling

In a shared-parameter linear model, a standard Thompson decision samples **one** parameter vector,

$$
\widetilde\theta_t\sim\mathcal N(\widehat\theta_t,v^2A_t^{-1}),
$$

then scores every candidate with the same draw, $x^\top\widetilde\theta_t$. The scale $v$ depends on the chosen posterior or exploration model.

The archived implementation samples inside the candidate-arm loop using covariance $A_t^{-1}$. It therefore draws a different parameter vector for each candidate. That changes the joint distribution of arm scores and does not implement the shared-draw rule above.

| Archived method label | Reported regret |
| --- | ---: |
| Linear Thompson Sampling | 1098.24 |

<details>
<summary>View the archived linear sampling curves</summary>
<div class="study-figure-grid">
<figure><img src="/online-learning/img/LinTS.png" alt="Per-arm parameter sampling: cumulative regret"><figcaption>Per-arm parameter sampling: cumulative regret</figcaption></figure>
<figure><img src="/online-learning/img/LinTS_est.png" alt="Per-arm parameter sampling: parameter error"><figcaption>Per-arm parameter sampling: parameter error</figcaption></figure>
</div>
</details>

The figure remains useful as a record of that implementation. Its score should not be treated as a benchmark of the standard shared-parameter algorithm.

## A nonlinear squared-reward experiment

The nonlinear simulator generates rewards of the form

$$
r=(x^\top\theta)^2+\epsilon,\qquad
\epsilon\sim\mathcal N(0,\sigma^2).
$$

The archived `GeneralizedLinearBandit` class still estimates $\widehat\theta=A^{-1}b$ from linear sufficient statistics, then squares the prediction and adds a linear confidence bonus:

$$
(x^\top\widehat\theta)^2+\alpha\sqrt{x^\top A^{-1}x}.
$$

This is a **squared-prediction heuristic**, not a validated GLM-UCB implementation. In particular, $A^{-1}b$ is not the maximum-likelihood estimator for this nonlinear observation model. Under the stated Gaussian noise model, an unregularized maximum-likelihood fit instead minimizes

$$
\sum_s\left[r_s-(x_s^\top\theta)^2\right]^2.
$$

That objective is generally nonlinear and nonconvex. The squared link is not globally monotone and cannot distinguish $\theta$ from $-\theta$, so parameter error against only one sign also needs care. Standard generalized-linear-bandit guarantees cannot simply be transferred to this experiment.

| Squared-prediction heuristic setting | Reported regret |
| --- | ---: |
| $\alpha=0.1$ | 62.16 |
| $\alpha=0.5$ | 727.63 |
| $\alpha=1.5$ | 5948.48 |

<details>
<summary>View the squared-reward experiment curves</summary>
<div class="study-figure-grid">
<figure><img src="/online-learning/img/GLMUCB01.png" alt="Squared-reward heuristic regret, alpha = 0.1"><figcaption>Squared-reward heuristic regret, alpha = 0.1</figcaption></figure>
<figure><img src="/online-learning/img/GLMUCB05.png" alt="Squared-reward heuristic regret, alpha = 0.5"><figcaption>Squared-reward heuristic regret, alpha = 0.5</figcaption></figure>
<figure><img src="/online-learning/img/GLMUCB15.png" alt="Squared-reward heuristic regret, alpha = 1.5"><figcaption>Squared-reward heuristic regret, alpha = 1.5</figcaption></figure>
<figure><img src="/online-learning/img/GLMUCB01_est.png" alt="Squared-reward heuristic parameter error, alpha = 0.1"><figcaption>Squared-reward heuristic parameter error, alpha = 0.1</figcaption></figure>
<figure><img src="/online-learning/img/GLMUCB05_est.png" alt="Squared-reward heuristic parameter error, alpha = 0.5"><figcaption>Squared-reward heuristic parameter error, alpha = 0.5</figcaption></figure>
<figure><img src="/online-learning/img/GLMUCB15_est.png" alt="Squared-reward heuristic parameter error, alpha = 1.5"><figcaption>Squared-reward heuristic parameter error, alpha = 1.5</figcaption></figure>
</div>
</details>

The original plots retain their historical “GLM-UCB” labels for traceability; the label does not resolve the estimator mismatch.

## Reproduction notes

Use the repository's `main` branch. [`Simulation.py`](https://github.com/kapshaul/OnlineLearning/blob/main/Simulation.py) selects independent-arm and linear methods; [`SimulationNonLinear.py`](https://github.com/kapshaul/OnlineLearning/blob/main/SimulationNonLinear.py) selects the nonlinear experiment. Algorithms are enabled in each script's `algorithms` dictionary.

Before making a new cross-algorithm comparison:

- Fix environment settings and use matched random seeds across repeated runs.
- Specify the actual sampling distribution, estimator, and confidence rule.
- Record the full-horizon total in addition to periodic checkpoints.
- Report a mean and uncertainty across runs, separately for each reward model.

No training or simulation results were regenerated during this content review. The external repository's implementations remain unchanged.

## Takeaways

- Exploration settings can dominate a single run; one regret value does not determine an algorithm ranking.
- Standard deviation, covariance, and shared parameter draws change the algorithm being evaluated.
- A nonlinear prediction formula does not make a linear estimator a nonlinear maximum-likelihood estimator.
- Reading the implementation is necessary to interpret the experiment's labels and metrics.

## Sources

- [Original bandit comparison experiment](https://github.com/kapshaul/OnlineLearning/blob/main/docs/bandits-comparison.md)
- [Primary simulation script](https://github.com/kapshaul/OnlineLearning/blob/main/Simulation.py)
- [NumPy normal distribution: the scale parameter](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html)
- [Parametric Bandits: The Generalized Linear Case](https://proceedings.neurips.cc/paper/2010/hash/c2626d850c80ea07e7511bbae4c76f4b-Abstract.html)
