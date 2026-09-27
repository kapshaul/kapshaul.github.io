---
title: "Word Embeddings: PPMI, GloVe & Bias"
date: 2024-04-10
lastmod: 2026-09-27
category: "NLP"
tags: ["Word Embeddings", "GloVe", "PPMI"]
author: ["Yong-Hwan Lee"]
summary: "Build count-based and learned word vectors, derive GloVe gradients, and interpret embedding visualizations and analogy results."
editPost:
  URL: "https://github.com/kapshaul/NLP-WordVector"
  Text: "GitHub"
---

## Overview

This study uses AG News text to compare positive pointwise mutual information (PPMI) with dimensionality reduction and learned GloVe vectors. A separate analysis examines analogy outputs from pretrained word2vec embeddings.

The charts, training log, and analogy scores below come from the original experiment. They are not newly measured results.

## Vocabulary and memory

The original report retained tokens occurring at least 12 times and reported approximately 96% token coverage. Coverage means the fraction of token occurrences represented, not the fraction of distinct word types.

<figure><img src="/word-vector/Figure_1.png" alt="Token frequency distribution and cumulative token coverage in the AG News vocabulary" /><figcaption>Vocabulary frequency and coverage in the recorded experiment.</figcaption></figure>

A dense co-occurrence matrix with vocabulary size $V$ requires $V^2b$ bytes, where $b$ is the number of bytes per entry. The earlier estimate of about 1 GB applies to that experiment's vocabulary and representation; it is not a general memory bound. Probability calculations and factorization can require additional copies.

## Count-based vectors with PPMI

Let $C_{ij}$ count occurrences of word $i$ with context $j$, and let $N=\sum_{i,j}C_{ij}$. Estimate

$$
P(i,j)=\frac{C_{ij}}{N},\quad
P(i)=\frac{\sum_j C_{ij}}{N},\quad
P(j)=\frac{\sum_i C_{ij}}{N}.
$$

For positive joint and marginal probabilities,

$$
\operatorname{PPMI}(i,j)=
\max\left(0,\log\frac{P(i,j)}{P(i)P(j)}\right).
$$

PPMI stands for **positive pointwise mutual information**. Zero-count pairs need explicit handling to avoid evaluating $\log 0$.

The repository computes $M\approx U_k\Sigma_kV_k^\top$, concatenates $U_k\Sigma_k^{1/2}$ and $V_k\Sigma_k^{1/2}$, and normalizes each row. With its default $k=16$, this produces 32-dimensional vectors before visualization.

<figure><img src="/word-vector/Figure_2.png" alt="Two-dimensional t-SNE projection of count-based word embeddings" /><figcaption>t-SNE projection of the reduced PPMI vectors.</figcaption></figure>

<details>
<summary>View the three topic close-ups</summary>
<figure><img src="/word-vector/Figure_3.png" alt="Close-up of war-related words in the embedding projection" /><figcaption>War-related words</figcaption></figure>
<figure><img src="/word-vector/Figure_4.png" alt="Close-up of technology-related words in the embedding projection" /><figcaption>Technology-related words</figcaption></figure>
<figure><img src="/word-vector/Figure_5.png" alt="Close-up of politics-related words in the embedding projection" /><figcaption>Politics-related words</figcaption></figure>
</details>

These views help inspect local neighborhoods. Distances between distant clusters, cluster sizes, and empty spaces in t-SNE should not be read as direct measurements of the original embedding geometry. See the [t-SNE paper](https://jmlr.org/papers/v9/vandermaaten08a.html).

## GloVe objective and gradients

For a pair with $C_{ij}>0$, define

$$
e_{ij}=\mathbf w_i^\top\widetilde{\mathbf w}_j+b_i+\widetilde b_j-\log C_{ij}.
$$

The objective and weighting function are

$$
J=\sum_{i,j:C_{ij}>0}f(C_{ij})e_{ij}^2,\qquad
f(x)=\min\left(1,\left(\frac{x}{x_{\max}}\right)^\alpha\right).
$$

The implementation uses $x_{\max}=100$ and $\alpha=0.75$. Zero-count pairs are omitted. This weighted regression objective follows the [GloVe paper](https://nlp.stanford.edu/pubs/glove.pdf).

### One pair versus the full objective

For the **single-pair loss** $\ell_{ij}=f(C_{ij})e_{ij}^2$,

$$
\begin{aligned}
\nabla_{\mathbf w_i}\ell_{ij}&=2f(C_{ij})e_{ij}\widetilde{\mathbf w}_j,\\
\nabla_{\widetilde{\mathbf w}_j}\ell_{ij}&=2f(C_{ij})e_{ij}\mathbf w_i,\\
\frac{\partial\ell_{ij}}{\partial b_i}
&=\frac{\partial\ell_{ij}}{\partial\widetilde b_j}
=2f(C_{ij})e_{ij}.
\end{aligned}
$$

The full gradient must sum every contribution involving the parameter:

$$
\nabla_{\mathbf w_i}J
=\sum_{j:C_{ij}>0}2f(C_{ij})e_{ij}\widetilde{\mathbf w}_j.
$$

The context-vector gradient sums over $i$; bias gradients sum over their respective partners. A minibatch implementation must accumulate repeated indices rather than overwrite their contributions.

### Recorded training behavior

The final logged averages over 100 batches were approximately 0.0467–0.0483. This describes the end of the run; a short, stable loss segment alone does not establish convergence or downstream quality.

```text
Iter 14400 / 15227: average loss = 0.046686563985831216
Iter 14700 / 15227: average loss = 0.04827717854832922
Iter 15200 / 15227: average loss = 0.04732485846561704
```

## Interpreting analogy results

The separate word2vec analysis queried $\mathbf v_b-\mathbf v_a+\mathbf v_c$. The original nearest-neighbor outputs included:

| Query | Selected returned words | Similarity scores |
|---|---|---|
| man : doctor :: woman : ? | gynecologist, nurse, doctors | 0.709, 0.648, 0.647 |
| woman : doctor :: man : ? | physician, doctors, surgeon | 0.646, 0.586, 0.572 |

These asymmetric associations motivate a broader bias evaluation. Scores measure vector similarity, not probabilities or facts about people. Two queries do not establish bias prevalence across the vocabulary or identify its cause. These pretrained word2vec results are separate from the AG News GloVe experiment.

## Reproducing the study

The [repository](https://github.com/kapshaul/NLP-WordVector) contains:

| Script | Purpose |
|---|---|
| `build_freq_vectors.py` | Counts, PPMI, SVD, and visualization |
| `build_glove_vectors.py` | GloVe training |
| `Exploring_learned_biases.py` | Pretrained embedding analogy analysis |

Record dependency versions, dataset, tokenizer, vocabulary cutoff, context definition, and random seed when comparing runs. The historical results have not been reproduced in a fresh training run.

## Takeaways

- Coverage and memory depend on tokenization, cutoff, and storage format.
- Distinguish a pairwise gradient from the full gradient when implementing GloVe.
- Visualizations and analogy queries support exploration; general claims need broader evaluation.
