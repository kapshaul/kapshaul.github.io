---
title: "Online Learning - Discounted UCB" 
date: 2023-12-03
lastmod: 2023-12-03
tags: ["Linear Regression","Online Learning","Recommendation System","UCB"]
author: ["Yong-Hwan Lee"]
description: "This study was carried out as a project at Oregon State University."
summary: "The focus of this study is to report the performance of Upper Confidence Bound (UCB) and discounted UCB with different choices of the discount factor (γ)."
editPost:
    URL: "https://github.com/kapshaul/OnlineLearning/tree/discountedUCB"
    Text: "GitHub"
showToc: true
disableAnchoredHeadings: false

---

## Overview
The focus of this study is to report the performance of Upper Confidence Bound (UCB) and discounted UCB with different choices of the discount factor (γ).

#### 1. Theoretical Aspect of the Effect of γ

In discounted UCB, recent and previous observations are weighted differently based on the discount factor γ. The weight for an observation decreases as it gets further in the past. This is represented by the term γ<sup>t−j</sup>, which is large when j is close to t, meaning recent observations significantly affect the estimate μ̂. Conversely, past observations far from the present time sequence t have minimal impact.

- **γ close to 1**: Minimal discounting of past observations, equivalent to standard UCB.
- **γ close to 0**: Significant discounting, primarily considering only the most recent observation.

When the environment changes suddenly after a time step T/2, relying on previous observations becomes less effective. Thus, a properly chosen γ in discounted UCB can outperform standard UCB by reducing dependence on outdated observations.

#### 2. Empirical Aspect of the Effect of γ

Empirical results show that:

- UCB exhibits the highest cumulative regret.
- Discounted UCB generally shows lower regret.
- The performance difference between γ = 0.1 and γ = 0.5 is not substantial.
- γ = 0.9 leads to higher cumulative regrets compared to the other two values of γ.

Given the sudden change in the reward function after T/2, past observations become less relevant, making discounted UCB a better choice than UCB. However, γ = 0.9 might be too large as it overly relies on past observations.

## Implementation

1. Clone the repository.
2. To run the simulation, execute the `Simulation.py` script.

## Results

#### Cumulative Regret of UCB and Discounted UCB

The table below summarizes the cumulative regrets for UCB and discounted UCB with different γ values across multiple trials.

<div align="center">

| Trial | UCB | DUCB (γ = 0.1) | DUCB (γ = 0.5) | DUCB (γ = 0.9) |
|-------|:---:|:--------------:|:--------------:|:--------------:|
| 1     | 11  |  3             | 4              | 6              |
| 2     | 37  |  3             | 2              | 6              |
| 3     | 59  |  4             | 2              | 7              |
| 4     | 8   |  3             | 2              | 3              |
| 5     | 12  |  2             | 2              | 7              |

</div>

#### Cumulative Regret Plots

<div align="center">

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/discount-ucb/UCB1.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/UCB2.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/UCB3.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/UCB4.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/UCB5.png" width="230">
  </div>

</div>

**Figure 1**: Cumulative regrets of UCB

<br>
<br>
<br>

(a) γ = 0.1

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB1_01.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB2_01.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB3_01.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB4_01.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB5_01.png" width="230">
  </div>

</div>

<br>
<br>

(b) γ = 0.5

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB1_05.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB2_05.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB3_05.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB4_05.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB5_05.png" width="230">
  </div>

</div>

<br>
<br>

(c) γ = 0.9

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB1_09.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB2_09.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB3_09.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB4_09.png" width="230">
  </div>

  <div style="text-align: center;">
    <img src="/discount-ucb/DUCB5_09.png" width="230">
  </div>

</div>

**Figure 2**: Cumulative regret of discounted UCB with different γ values

</div>

## Conclusion

The results suggest that discounted UCB with a properly chosen γ can significantly reduce cumulative regret compared to standard UCB, especially in environments with sudden changes.
