---
title: "Discounted UCB in a Changing Environment"
date: 2023-12-03
lastmod: 2026-09-29
category: "Online Learning"
tags: ["Nonstationary Bandits", "UCB", "Discounting"]
author: ["Yong-Hwan Lee"]
summary: "Study how forgetting old rewards helps after a change point, and distinguish time-based discounting from the selected-arm update used in the archived experiment."
editPost:
    URL: "https://github.com/kapshaul/online-learning/blob/main/docs/discounted-ucb.md"
    Text: "GitHub"
---

## Overview

When reward distributions change, a long history of observations can slow adaptation. This study compares UCB with an exponentially weighted variant in a two-arm environment whose best arm switches halfway through a run.

The archived code discounts only the selected arm. This differs from standard elapsed-time discounting in discounted UCB. The distinction is central to interpreting the results below.

## What time-based discounting means

Let $A_s$ denote the selected arm at step $s$, $r_s$ its observed reward, and $0<\gamma\leq1$ the discount factor. After step $t$, define

$$
N_t(i)=\sum_{s=1}^{t}\gamma^{t-s}\mathbf1\{A_s=i\},
\qquad
S_t(i)=\sum_{s=1}^{t}\gamma^{t-s}r_s\mathbf1\{A_s=i\}.
$$

For $N_t(i)>0$, the discounted mean is $\widehat\mu_t(i)=S_t(i)/N_t(i)$. An arm's older observations lose weight with **every elapsed round**, whether or not that arm is selected.

The following dependency-free example updates these statistics. It is not a complete policy: action selection, initialization, and the confidence bonus must also be specified.

```python
def update_discounted_statistics(counts, sums, arm, reward, gamma):
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if len(counts) != len(sums) or not 0 <= arm < len(counts):
        raise ValueError("counts, sums, and arm must describe the same arm set")
    next_counts = [gamma * count for count in counts]
    next_sums = [gamma * total for total in sums]
    next_counts[arm] += 1
    next_sums[arm] += reward
    return next_counts, next_sums
```

Discounting both a sum and its count leaves an unselected arm's mean unchanged, but reduces its effective evidence. A discounted confidence rule can then encourage revisiting it.

| Discount factor | Effect on the statistics |
| --- | --- |
| $\gamma=1$ | No discounting; ordinary counts and reward sums |
| $\gamma$ close to 1 | Slow forgetting, with finite accumulated weight when $\gamma<1$ |
| Smaller positive $\gamma$ | Faster forgetting and potentially noisier estimates |

For an arm selected every round, its count approaches $1/(1-\gamma)$ when $\gamma<1$. This is an approximate memory scale, not a guarantee of the best setting. “Close to one” is not mathematically equivalent to one.

## What the archived code actually does

In `lib/DiscountedUCBBandit.py`, the selected arm receives

$$
N_i\leftarrow\gamma N_i+1,\qquad
S_i\leftarrow\gamma S_i+r.
$$

All other arms retain their previous statistics. Thus an observation is discounted according to later **selections of the same arm**, not according to elapsed global time. The code's confidence bonus uses the undiscounted update counter:

$$
\widehat\mu_i+\sqrt{\frac{2\alpha\log t}{N_i}}.
$$

For example, after selecting arm 0 once and then arm 1 twice, the time-discounted count of arm 0 is $\gamma^2$. The archived implementation keeps that count at 1.

The experiment should therefore be read as a comparison with this selected-arm-discount variant. The illustrative update above explains the standard time-based statistics; it does not retroactively change the archived results or update the external repository.

## Experiment setup

The linked simulation configures:

| Setting | Value |
| --- | --- |
| Horizon | 1,000 iterations |
| Arms / users | 2 / 1 |
| First half: expected rewards | Arm 0 = 1, arm 1 = 0 |
| Second half: expected rewards | Arm 0 = 0, arm 1 = 1 |
| Observation noise | Gaussian, mean 0 and standard deviation 0.1 |
| Exploration parameter | $\alpha=0.5$ |

The comparator chooses the best arm at each step, including after the switch. Regret uses expected reward gaps: the simulator adds the same noise to the selected and optimal rewards, so the noise cancels in their difference. In this binary-mean environment, each suboptimal action contributes one unit.

The script logs at zero-based iterations 0, 100, …, 900, then prints the last logged value. That value covers 901 actions, not all 1,000. The source needs an explicit final checkpoint before its printed output can be called full-horizon regret.

## Recorded results

The five-trial table is preserved from the original report. The mean row is the arithmetic mean of the displayed values. Seeds and per-run logs were not preserved alongside the table, so these values have not been re-established as full-horizon totals.

| Trial | UCB | Variant: $\gamma=0.1$ | Variant: $\gamma=0.5$ | Variant: $\gamma=0.9$ |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11 | 3 | 4 | 6 |
| 2 | 37 | 3 | 2 | 6 |
| 3 | 59 | 4 | 2 | 7 |
| 4 | 8 | 3 | 2 | 3 |
| 5 | 12 | 2 | 2 | 7 |
| **Mean** | **25.4** | **3.0** | **2.4** | **5.8** |

Within these recorded values, all three discounted variants have lower means than UCB. The $\gamma=0.5$ setting has the lowest mean, but five trials without matched seeds or uncertainty estimates do not establish a broadly optimal discount factor. These runs also do not measure the standard time-discounted algorithm.

<details>
<summary>UCB: five recorded trials</summary>
<div class="study-figure-grid">
<figure><img src="/discount-ucb/UCB1.png" alt="UCB cumulative regret, trial 1"><figcaption>Trial 1</figcaption></figure>
<figure><img src="/discount-ucb/UCB2.png" alt="UCB cumulative regret, trial 2"><figcaption>Trial 2</figcaption></figure>
<figure><img src="/discount-ucb/UCB3.png" alt="UCB cumulative regret, trial 3"><figcaption>Trial 3</figcaption></figure>
<figure><img src="/discount-ucb/UCB4.png" alt="UCB cumulative regret, trial 4"><figcaption>Trial 4</figcaption></figure>
<figure><img src="/discount-ucb/UCB5.png" alt="UCB cumulative regret, trial 5"><figcaption>Trial 5</figcaption></figure>
</div>
</details>

<details>
<summary>Discounted variant, gamma = 0.1: five recorded trials</summary>
<div class="study-figure-grid">
<figure><img src="/discount-ucb/DUCB1_01.png" alt="Discounted variant, gamma = 0.1 cumulative regret, trial 1"><figcaption>Trial 1</figcaption></figure>
<figure><img src="/discount-ucb/DUCB2_01.png" alt="Discounted variant, gamma = 0.1 cumulative regret, trial 2"><figcaption>Trial 2</figcaption></figure>
<figure><img src="/discount-ucb/DUCB3_01.png" alt="Discounted variant, gamma = 0.1 cumulative regret, trial 3"><figcaption>Trial 3</figcaption></figure>
<figure><img src="/discount-ucb/DUCB4_01.png" alt="Discounted variant, gamma = 0.1 cumulative regret, trial 4"><figcaption>Trial 4</figcaption></figure>
<figure><img src="/discount-ucb/DUCB5_01.png" alt="Discounted variant, gamma = 0.1 cumulative regret, trial 5"><figcaption>Trial 5</figcaption></figure>
</div>
</details>

<details>
<summary>Discounted variant, gamma = 0.5: five recorded trials</summary>
<div class="study-figure-grid">
<figure><img src="/discount-ucb/DUCB1_05.png" alt="Discounted variant, gamma = 0.5 cumulative regret, trial 1"><figcaption>Trial 1</figcaption></figure>
<figure><img src="/discount-ucb/DUCB2_05.png" alt="Discounted variant, gamma = 0.5 cumulative regret, trial 2"><figcaption>Trial 2</figcaption></figure>
<figure><img src="/discount-ucb/DUCB3_05.png" alt="Discounted variant, gamma = 0.5 cumulative regret, trial 3"><figcaption>Trial 3</figcaption></figure>
<figure><img src="/discount-ucb/DUCB4_05.png" alt="Discounted variant, gamma = 0.5 cumulative regret, trial 4"><figcaption>Trial 4</figcaption></figure>
<figure><img src="/discount-ucb/DUCB5_05.png" alt="Discounted variant, gamma = 0.5 cumulative regret, trial 5"><figcaption>Trial 5</figcaption></figure>
</div>
</details>

<details>
<summary>Discounted variant, gamma = 0.9: five recorded trials</summary>
<div class="study-figure-grid">
<figure><img src="/discount-ucb/DUCB1_09.png" alt="Discounted variant, gamma = 0.9 cumulative regret, trial 1"><figcaption>Trial 1</figcaption></figure>
<figure><img src="/discount-ucb/DUCB2_09.png" alt="Discounted variant, gamma = 0.9 cumulative regret, trial 2"><figcaption>Trial 2</figcaption></figure>
<figure><img src="/discount-ucb/DUCB3_09.png" alt="Discounted variant, gamma = 0.9 cumulative regret, trial 3"><figcaption>Trial 3</figcaption></figure>
<figure><img src="/discount-ucb/DUCB4_09.png" alt="Discounted variant, gamma = 0.9 cumulative regret, trial 4"><figcaption>Trial 4</figcaption></figure>
<figure><img src="/discount-ucb/DUCB5_09.png" alt="Discounted variant, gamma = 0.9 cumulative regret, trial 5"><figcaption>Trial 5</figcaption></figure>
</div>
</details>

## Reproduction notes

Use the repository's `main` branch and [`SimulationDiscountedUCB.py`](https://github.com/kapshaul/online-learning/blob/main/SimulationDiscountedUCB.py), which preserves the archived experiment's simulation script unchanged. The algorithm dictionary selects UCB or `DiscountedUCBBandit` and sets $\alpha$ and $\gamma$.

A revised experiment should distinguish the archived variant from a time-discounted implementation, append the final-horizon checkpoint, and use matched seeds across methods. Testing multiple change points, reward gaps, and stationary periods would show whether faster forgetting helps consistently.

The curves above remain the original archive. No simulations were rerun for this content revision.

## Takeaways

- Forgetting by elapsed time and forgetting by arm selections are different algorithms.
- Smaller discount factors can improve responsiveness while increasing estimation noise.
- A performance table must identify the implemented policy, comparator, and recorded horizon.
- These archived trials motivate a controlled rerun; they do not validate a universal ranking.

## Sources

- [Original discounted-bandit experiment](https://github.com/kapshaul/online-learning/blob/main/docs/discounted-ucb.md)
- [On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems](https://arxiv.org/abs/0805.3415)
