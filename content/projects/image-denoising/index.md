---
title: "Poisson Reconstruction for Medical Imaging"
date: 2019-12-10
lastmod: 2026-09-29
tags: ["EM Algorithm","Poisson Model","MLE","CRLB","Monte Carlo Simulation"]
author: ["Yong-Hwan Lee","Tony Storey"]
description: "This study was carried out as a project at Oregon State University."
summary: "Expectation maximization for a nine-parameter linear Poisson inverse problem, with Fisher-information analysis and a seeded 40,000-trial synthetic tomography study."
cover:
    image: "image.png"
    alt: "Illustrative CT image from the original course project, not an experimental result"
    relative: false

---

---

## Download

+ [Document](paper.pdf)
+ [LaTeX source](report.tex)
+ [Code](https://github.com/kapshaul/ct-medical-imaging)

---

## Overview

Photon-counting measurements connect a physical forward model to a statistical estimator. This project studies a deliberately small version of that problem: recover nine nonnegative intensities on a $3 \times 3$ grid from 16 independent Poisson counts. The report derives the likelihood, the Fisher information, and a nonnegative expectation-maximization (EM) update, then checks them in a seeded synthetic study.

The model is educational and emission-style: each count's mean is an additive sum of pixel intensities. It is not the exponential attenuation model of transmission X-ray CT. One lesson runs through the results: raising the likelihood of the observed counts and recovering the true field accurately are different goals.

## Measurement model

### Linear Poisson counts

With $m = 16$ observations and $n = 9$ unknown intensities $x \ge 0$,

$$
Y_i \sim \operatorname{Poisson}(\mu_i), \qquad \mu_i = (Ax)_i = \sum_{j=1}^{9} a_{ij} x_j, \qquad i = 1, \dots, 16.
$$

The known matrix $A \in \mathbb{R}_+^{16 \times 9}$ has binary entries that mark which pixels contribute to each count. It is an idealized sensing design, not a calibrated scanner geometry. Identifiability depends on full column rank, and the fixed matrix used here has rank 9.

For comparison, an ideal transmission CT measurement has mean $I_{0,i}\exp(-\sum_j l_{ij}\alpha_j) + r_i$, where $\alpha_j$ is an attenuation coefficient [3]. Taking a logarithm linearizes the background-free noiseless relation but does not preserve the Poisson distribution. So $x_j$ here is an intensity, not an attenuation coefficient, and the conclusions apply to the additive model only.

### From pixels to a measurement

Each voxel $p_j$ ($j = 1, \dots, 9$, numbered row by row) holds an unknown intensity $x_j$. The diagram below shows six of the 16 rows of the fixed matrix $A$: the three horizontal paths $Y_1$ to $Y_3$ and the three vertical paths $Y_9$ to $Y_{11}$. Every crossed pixel contributes a hidden count $N_{ij}$ with mean $a_{ij} x_j$, and pixels off the path have $a_{ij} = 0$. The remaining rows cover diagonal and single-pixel paths. Select a row or column to trace its three contributing voxels.

<div class="voxel-projection"></div>

## Likelihood and information

### Log-likelihood and I-divergence

Independence gives the likelihood of an observed count vector $y$:

$$
\begin{aligned}
L(x; y) &= \prod_{i=1}^{16} \frac{e^{-(Ax)_i} (Ax)_i^{y_i}}{y_i!}, \\
\ell(x; y) &= \sum_{i=1}^{16} \Big[ y_i \log (Ax)_i - (Ax)_i - \log \Gamma(y_i + 1) \Big].
\end{aligned}
$$

The estimate maximizes $\ell$ over $x \ge 0$, with the convention $0 \log \mu = 0$ for zero counts. The generalized Kullback–Leibler divergence (I-divergence),

$$
D(y \,\Vert\, Ax) = \sum_{i} \left[ y_i \log \frac{y_i}{(Ax)_i} - y_i + (Ax)_i \right],
$$

differs from $-\ell(x; y)$ only by terms that do not depend on $x$. Minimizing it is therefore equivalent to maximizing the Poisson likelihood. Neither vector needs to sum to one.

### Fisher information and the Cramér–Rao reference

At positive means, the expected negative Hessian of $\ell$ is

$$
\mathcal{I}(x) = A^\mathsf{T} \operatorname{diag}\!\left( \frac{1}{(Ax)_i} \right) A,
\qquad \operatorname{Cov}(\hat{x}) \succeq \mathcal{I}(x)^{-1}.
$$

The covariance bound holds for unbiased estimators under the usual regularity conditions, and the study evaluates $\mathcal{I}$ at the true $x$. A nonnegative, finite-iteration EM estimator may be biased. For it, the CRLB is a reference point, not a guaranteed lower bound on MSE.

For $x = gb$ with gain $g$ and fixed base field $b$, $\mathcal{I}(gb) = \mathcal{I}(b)/g$. Errors are compared on the base scale, $\hat{b} = \hat{x}/g$, so the reference is normalized by the actual $g^2$:

$$
C_b(g) = \frac{1}{n g^2} \operatorname{tr}\!\left[ \mathcal{I}(gb)^{-1} \right] = \frac{1}{n g} \operatorname{tr}\!\left[ \mathcal{I}(b)^{-1} \right].
$$

This normalized information reference falls as $1/g$.

## Expectation maximization

### Latent counts and the E step

Introduce independent latent counts $N_{ij} \sim \operatorname{Poisson}(a_{ij} x_j)$ with $Y_i = \sum_j N_{ij}$. Given the total $Y_i = y_i$, the contributions are multinomial with probabilities $a_{ij} x_j^{(t)} / (Ax^{(t)})_i$, so

$$
\hat{N}_{ij}^{(t)} = \mathbb{E}\big[ N_{ij} \mid y_i, x^{(t)} \big] = y_i \, \frac{a_{ij} x_j^{(t)}}{(Ax^{(t)})_i}.
$$

### M step and the multiplicative update

The expected complete-data log-likelihood separates by pixel. Entries with $a_{ij}=0$ have zero latent counts and contribute zero:

$$
Q(x \mid x^{(t)}) = \sum_{i,j} \Big[ \hat{N}_{ij}^{(t)} \log (a_{ij} x_j) - a_{ij} x_j \Big] + C.
$$

Setting $\partial Q / \partial x_j = 0$ gives the classical Poisson reconstruction update [2]:

$$
x^{(t+1)} = x^{(t)} \odot \frac{A^\mathsf{T} \big( y / Ax^{(t)} \big)}{A^\mathsf{T} \mathbf{1}},
$$

with elementwise division and $\odot$ denoting elementwise multiplication. From a valid start, each update leaves the observed likelihood nondecreasing [1]. Every coordinate must start strictly positive, since a zero coordinate stays at zero.

## Seeded simulation study

### Setup

| Setting | Value |
| --- | --- |
| Sensing matrix | Fixed predefined $16 \times 9$ binary matrix, rank 9 (no random model) |
| Base field $b$ | $(120, 240, 360, 180, 720, 300, 90, 420, 540)$ |
| Gains | $g \in \{0.1, 1, 5, 10\}$, true intensity $x = gb$, $Y \sim \operatorname{Poisson}(Agb)$ |
| Monte Carlo size | 10,000 trials per gain, 40,000 total |
| Random generator | NumPy PCG64, seed 20260929 |
| Initialization | $x^{(0)} = \max(\operatorname{lstsq}(A, y), 10^{-8})$ elementwise |
| Iterations | 1,000 EM updates, checkpoints at 0, 20, 200, and 1,000 |
| CRLB | $\mathcal{I}(gb)$ evaluated at the true intensity |

For trial $r$, the error is $q_r = \lVert \hat{x}_r / g - b \rVert_2^2 / 9$. The reported MSE is the mean over $R = 10{,}000$ trials, with approximate 95% Monte Carlo intervals $\bar{q} \pm 1.96\, s_q / \sqrt{R}$. These intervals describe uncertainty in the Monte Carlo average, not confidence in any individual reconstructed pixel.

### Error across gain

After 1,000 updates, with MSE and CRLB divided by $g^2$:

| Gain | MSE / $g^2$ | 95% MC interval | CRLB / $g^2$ | MSE / CRLB |
| ---: | ---: | :---: | ---: | ---: |
| 0.1 | 2,106.72 | [2,084.11, 2,129.34] | 2,110.20 | 0.9984 |
| 1 | 210.66 | [208.43, 212.89] | 211.02 | 0.9983 |
| 5 | 42.49 | [42.05, 42.94] | 42.20 | 1.0069 |
| 10 | 21.16 | [20.94, 21.39] | 21.10 | 1.0029 |

The normalized MSE follows the predicted $1/g$ scale. At every gain, EM lowers the average error by about 17% relative to its clipped least-squares start, for example from 254.30 to 210.66 at $g = 1$. Each interval covers the CRLB, so the results sit near the reference within Monte Carlo uncertainty. That agreement does not prove efficiency: unbiasedness has not been shown for this constrained estimator. The comparison also holds one matrix and one field fixed.

### Likelihood is not reconstruction error

The first trial at $g = 1$ separates optimization from estimation. Its log-likelihood rises monotonically from $-68.837$ to $-67.373$. Its MSE, however, first increases from 36.07 to about 37.9 over the first three updates, then falls to 33.17 by update 1,000. Better data fit does not guarantee a closer estimate of the truth.

### Stopping diagnostics

With the relative iterate change $\delta_t = \lVert x^{(t)} - x^{(t-1)} \rVert_2 / \lVert x^{(t-1)} \rVert_2$, at most 0.42% of trials meet $\delta_{20} < 10^{-6}$ at any gain. At 200 updates, all do. The average MSE, however, is already near its final value after 20 updates (210.66 at both checkpoints for $g = 1$). At $g = 0.1$, 2.57% of least-squares starts had a negative coordinate, which the positivity floor corrects. Across 40 million trial-level updates, the largest floating-point likelihood decrease was $1.62 \times 10^{-10}$, within the stated numerical tolerance.

## Reproducing the results

The study script and its outputs are available here: [reproduce.py](report/scripts/reproduce.py), [results.json](report/data/results.json), [summary.csv](report/data/summary.csv), and [trace.csv](report/data/trace.csv). If you download them, keep the `report/scripts/` and `report/data/` folder arrangement: the script writes its outputs to the `data` folder beside `scripts`. From the folder that contains `report/` (the repository root), run with Python 3 and NumPy (the recorded run used NumPy 2.3.5):

```bash
python3 report/scripts/reproduce.py --trials 10000 --iterations 1000 --seed 20260929
```

These flags equal the script's defaults, so `python3 report/scripts/reproduce.py` alone reproduces the recorded run. The script also checks likelihood monotonicity, a noise-free fixed point, and scalar-versus-vectorized updates. A different NumPy version may change the last displayed digits.

The original MATLAB code in the Code repository is unchanged and does not produce these numbers. It overwrites the fixed matrix with an unseeded random one, divides MSE by the squared gain index rather than $g^2$, evaluates the Fisher information at the current estimate, and starts EM without a positivity floor. It uses the same 10,000 trials per gain and four gain values, but only 20 EM updates. The historical PDF describes yet another setup: 200 trials and a gain sweep from 0.1 to 100.

## Scope

All evidence on this page comes from synthetic data under the model above. It verifies a small statistical inverse problem. It does not establish diagnostic performance, dose reduction, or image quality at clinical resolution. Extending the analysis to transmission CT would require the exponential forward model, calibrated scanner geometry, and background and electronic noise. At low counts, this can call for mixed Poisson–Gaussian modeling [4].

## References

1. A. P. Dempster, N. M. Laird, and D. B. Rubin, "Maximum likelihood from incomplete data via the EM algorithm," *Journal of the Royal Statistical Society, Series B*, 39(1), 1–22, 1977. [doi:10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)
2. L. A. Shepp and Y. Vardi, "Maximum likelihood reconstruction for emission tomography," *IEEE Transactions on Medical Imaging*, 1(2), 113–122, 1982. [doi:10.1109/TMI.1982.4307558](https://doi.org/10.1109/TMI.1982.4307558)
3. E. A. Rashed and H. Kudo, "Towards high-resolution synchrotron radiation imaging with statistical iterative reconstruction," *Journal of Synchrotron Radiation*, 20(1), 116–124, 2013. [doi:10.1107/S0909049512041301](https://doi.org/10.1107/S0909049512041301)
4. Q. Ding, Y. Long, X. Zhang, and J. A. Fessler, "Statistical image reconstruction using mixed Poisson–Gaussian noise model for X-ray CT," arXiv:1801.09533, 2018. [arxiv.org/abs/1801.09533](https://arxiv.org/abs/1801.09533)
