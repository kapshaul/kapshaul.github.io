---
title: "Online Learning - Comparison of Online Learning Algorithms" 
date: 2023-10-20
lastmod: 2023-10-22
tags: ["Linear Regression","Online Learning","Adversarial Learning","Recommendation System","UCB"]
author: ["Yong-Hwan Lee"]
description: "This study was carried out as a project at Oregon State University."
summary: "Experiment with and compare online learning algorithms, including Bandits, UCB, and related methods."
editPost:
    URL: "https://github.com/kapshaul/OnlineLearning/tree/bandits-comparison-analysis"
    Text: "GitHub"
showToc: true
disableAnchoredHeadings: false

---

---

## Overview
This page includes implementations and performance reports of several online learning algorithms.

---

## Implementation
To simulate a specific algorithm, edit the `Simulation.py` script by enabling the desired algorithm and disabling the others.

For example, to run the UCB algorithm with $\alpha = 0.5$, update the code as follows:
```python
## Initiate Bandit Algorithms ##
algorithms = {}

#algorithms['EpsilonGreedyLinearBandit'] = EpsilonGreedyLinearBandit(dimension=context_dimension, lambda_=0.1, epsilon=None)
#algorithms['EpsilonGreedyMultiArmedBandit'] = EpsilonGreedyMultiArmedBandit(num_arm=n_articles, epsilon=0.1)
#algorithms['ExplorethenCommit'] = ExplorethenCommit(num_arm=n_articles, m=30)
algorithms['UCBBandit'] = UCBBandit(num_arm=n_articles, alpha=0.5)
#algorithms['ThompsonSamplingGaussianMAB'] = ThompsonSamplingGaussianMAB(num_arm=n_articles)
#algorithms['LinearUCBBandit'] = LinearUCBBandit(dimension=context_dimension, lambda_=0.1, alpha=0.5) #delta=0.05, alpha=2.358
#algorithms['LinearThompsonSamplingMAB'] = LinearThompsonSamplingMAB(dimension=context_dimension, lambda_=0.1)
```

After selecting your algorithm, run the `Simulation.py` script.

---

## 1. Explore-then-Commit

#### Result

<div align="center">

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/online-learning/img/ETC10.png" width="230">
    <div>(a) \( m = 10 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/ETC20.png" width="230">
    <div>(b) \( m = 20 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/ETC30.png" width="230">
    <div>(c) \( m = 30 \)</div>
  </div>
  
</div>

**Figure 1**: Explore then Commit accumulated regret

| Hyperparameter (m) | Cumulative Regret |
|:------------------:|:-----------------:|
|         10         |      1001.40      |
|         20         |       214.90      |
|         30         |       334.02      |

</div>

---

## 2. Upper Confidence Bound (UCB)

#### Reward Estimation + Confidence Bound

$$
\text{UCB} = \hat u_{t-1,i} + \sqrt{\frac{2 \ln t}{S_{t-1,i}}}
$$

#### Result

<div align="center">

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/online-learning/img/UCB01.png" width="230">
    <div>(a) \( \alpha = 0.1 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/UCB05.png" width="230">
    <div>(b) \( \alpha = 0.5 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/UCB1.png" width="230">
    <div>(c) \( \alpha = 1 \)</div>
  </div>

</div>

**Figure 2**: UCB Bandit accumulated regret

| Hyperparameter (α) | Cumulative Regret |
|:------------------:|:-----------------:|
| 0.1                | 256.50            |
| 0.5                | 977.03            |
| 1.0                | 1906.65           |

</div>

---

## 3. Thompson Sampling
#### Posterior Distribution

$$
N \sim \left( \hat u_{t-1,i}, \frac{1}{S_{t-1,i} + 1} \right)
$$

#### Result

<div align="center">

<img src="/online-learning/img/TS.png" width="350">

**Figure 3**: Thompson Sampling accumulated regret

| Cumulative Regret |
|:------------------:|
|  100              |

</div>

---

## 4. Linear UCB (LinUCB)

#### Parameter Estimation

$$
\hat \theta_{t+1} = A^{-1}_ {t+1} b_{t+1}
$$

#### Reward Estimation + Confidence Bound

$$
\text{UCB} = x^T \hat \theta_t + \alpha \sqrt{x^T A^{-1} x}
$$

### Result

<div align="center">

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/online-learning/img/LinUCB05.png" width="230">
    <div>(a) \( \alpha = 0.5 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/LinUCB15.png" width="230">
    <div>(b) \( \alpha = 1.5 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/LinUCB25.png" width="230">
    <div>(c) \( \alpha = 2.5 \)</div>
  </div>

</div>

**Figure 4**: Linear UCB accumulated regret

</div>

<br>

<div align="center">

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/online-learning/img/LinUCB05_est.png" width="230">
    <div>(a) \( \alpha = 0.5 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/LinUCB15_est.png" width="230">
    <div>(b) \( \alpha = 1.5 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/LinUCB25_est.png" width="230">
    <div>(c) \( \alpha = 2.5 \)</div>
  </div>

</div>

**Figure 5**: Linear UCB estimation error

| Hyperparameter (α) | Cumulative Regret |
|:------------------:|:-----------------:|
| 0.5                | 24.43             |
| 1.5                | 177.89            |
| 2.5                | 487.73            |

</div>

---

## 5. Linear Thompson Sampling (LinTS)

#### Posterior Distribution

$$
N \sim (\hat{\theta}_t, A^{-1}_t)
$$

### Result

<div align="center">

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/online-learning/img/LinTS.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/LinTS_est.png" width="230">
  </div>

</div>

**Figure 6**: Linear Thompson Sampling accumulated regret and estimation error

| Cumulative Regret |
|:------------------:|
| 1098.24            |

</div>

---

## 6. Generalized Linear Model (GLM) Bandit: Non-linear Bandit

#### Modified Non-LinearReward Function For Testing

$$
R = (x^T \theta)^2 + \epsilon, \text{ where } \epsilon \sim N(\mu, \sigma^2)
$$

#### GLM Parameter Estimation (MLE)

$$
\hat \theta_{t+1} = \max_{\theta} P(r|\theta) = A^{-1}_ {t+1} b_{t+1}
$$

#### GLM UCB

$$
UCB_{GLM} = f(x^T \hat \theta_t) + \alpha \sqrt{x^T A^{-1} x} = (x^T \hat \theta_t)^2 + \alpha \sqrt{x^T A^{-1} x}
$$

#### Result

<div align="center">

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/online-learning/img/GLMUCB01.png" width="230">
    <div>(a) \( \alpha = 0.1 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/GLMUCB05.png" width="230">
    <div>(b) \( \alpha = 0.5 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/GLMUCB15.png" width="230">
    <div>(c) \( \alpha = 1.5 \)</div>
  </div>

</div>

**Figure 7**: GLM-UCB accumulated regret

<br>

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/online-learning/img/GLMUCB01_est.png" width="230">
    <div>(a) \( \alpha = 0.1 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/GLMUCB05_est.png" width="230">
    <div>(b) \( \alpha = 0.5 \)</div>
  </div>

  <div style="text-align: center;">
    <img src="/online-learning/img/GLMUCB15_est.png" width="230">
    <div>(c) \( \alpha = 1.5 \)</div>
  </div>

</div>

**Figure 8**: GLM-UCB estimation error

| Hyperparameter (α) | Cumulative Regret |
|:------------------:|:-----------------:|
| 0.1                | 62.16             |
| 0.5                | 727.63            |
| 1.5                | 5948.48           |

</div>
