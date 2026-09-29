---
title: "Language Model Decoding: Sampling & Beam Search"
date: 2024-03-12
lastmod: 2026-09-29
category: "NLP"
tags: ["Text Generation", "Sampling", "Beam Search"]
author: ["Yong-Hwan Lee"]
summary: "Compare temperature, top-k, nucleus sampling, and beam search, with attention to probability filtering, recurrent state, and reproducibility."
editPost:
    URL: "https://github.com/kapshaul/NLP-sampling.search/tree/main"
    Text: "GitHub"
---

## Overview

This Oregon State University study implements decoding strategies around a provided, pre-trained LSTM language model. The model has three recurrent layers, a hidden size of 512, and 100-dimensional token embeddings. Its checkpoint was trained on text from the first five *A Song of Ice and Fire* novels.

The contribution explored here is the decoder: how next-token probabilities are filtered or searched, and how those choices affect generated text. The examples are qualitative observations from the original experiment, not a benchmark of language quality.

## Sampling methods

Let $z_i$ be the logit of token $i$. For a strictly positive temperature $\tau$,

$$
p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}.
$$

| Method | Rule | Boundary cases |
| --- | --- | --- |
| Vanilla sampling | Draw from the full softmax distribution | Equivalent to temperature 1 with no filtering |
| Temperature | Divide logits by $\tau>0$ before softmax | Small $\tau$ concentrates mass near the largest logits; large $\tau$ flattens it |
| Top-k | Keep the $k$ highest-probability tokens and renormalize | $k=1$ is greedy decoding; require $1\leq k\leq V$ |
| Top-p / nucleus | Keep the smallest sorted prefix whose cumulative probability is at least $p$ | Require $0<p\leq1$; $p=1$ preserves the distribution |

Here $V$ is vocabulary size. Top-p must retain the token that crosses the probability threshold. Removing that token can leave less than the requested mass—or even an empty set when the highest-probability token alone exceeds $p$.

A very small positive temperature is not the same as setting temperature to zero, which is invalid in this formula. With tied maximum logits, the low-temperature limit can retain more than one candidate.

### A filtering example

This standalone helper illustrates the probability-filtering step for a one-dimensional vector of finite logits. It follows the original exercise's convention of selecting either top-k or top-p. Prompt processing and recurrent state management are separate.

```python
import torch


def next_token_probs(logits, temperature=1.0, k=0, p=1.0):
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("Expected a nonempty vocabulary vector")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("Expected finite floating-point logits")
    if not (0 < temperature < float("inf") and 0 < p <= 1):
        raise ValueError("Require finite temperature > 0 and 0 < p <= 1")
    if not isinstance(k, int) or not 0 <= k <= logits.numel():
        raise ValueError("k must be an integer between 0 and vocabulary size")
    if k and p != 1.0:
        raise ValueError("Choose top-k or top-p for this example")

    scores = (logits - logits.max()) / temperature
    if k:
        values, indices = torch.topk(scores, k)
        scores = torch.full_like(scores, -torch.inf).scatter(0, indices, values)
    elif p < 1.0:
        sorted_scores, order = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=0)
        # Remove a token only when earlier tokens already reach p.
        preceding_mass = torch.cat(
            (sorted_probs.new_zeros(1), sorted_probs.cumsum(0)[:-1])
        )
        sorted_scores = sorted_scores.masked_fill(preceding_mass >= p, -torch.inf)
        scores = torch.empty_like(scores).scatter(0, order, sorted_scores)

    return torch.softmax(scores, dim=0)
```

For probabilities `[0.60, 0.25, 0.15]` and `p=0.75`, the retained prefix is the first two tokens, with mass `0.85` before renormalization. Top-p changes candidate-set size with the distribution; it does not guarantee that every sampled continuation will be fluent.

### Recorded observations

The original experiment used the same opening prompt while varying the decoder.

| Configuration | Observation in the recorded examples |
| --- | --- |
| $\tau=0.0001$ | Similar continuation to greedy decoding |
| $\tau=100$ | Disconnected, largely incoherent token sequences |
| $k=1$ | One candidate per step |
| $k=20$ | More varied output, including awkward phrases and unknown tokens |
| $p=0.001$ | Greedy-like output in this example |
| $p=0.75$ | A wider range of continuations, with remaining grammatical errors |
| $p=1$ | Same sampling distribution as vanilla sampling |

Equal distributions do not imply identical sampled strings unless the random state and every other generation setting also match. A few examples cannot establish an optimal temperature, $k$, or $p$.

## Beam search

Beam search maintains up to $B$ candidate prefixes. At each step, it extends the candidates and keeps the highest-scoring prefixes according to accumulated log probability:

$$
s(y_{1:t})=\sum_{j=1}^{t}\log P(y_j\mid y_{<j},\text{prompt}).
$$

It approximates the search for a high-probability sequence. Finite-width pruning can discard the globally best completion, so the method is not an exact global optimizer.

1. Process the prompt once to obtain its recurrent state and next-token logits.
2. Expand each retained prefix and add the next token's log probability to its score.
3. Keep the best $B$ candidates together with the matching hidden and cell states.
4. Feed each new token with its associated state, then repeat.

The linked implementation loops over beams individually. It does **not** perform batched inference across all beams. It stops at a fixed maximum length and does not implement EOS-aware completion or length normalization; comparisons with decoders that do require those conventions to be aligned.

### Interpreting beam width

| Width | Interpretation |
| --- | --- |
| $B=1$ | Greedy decoding, provided prompt processing, scoring, and stopping rules match |
| Larger $B$ | Retains more search alternatives and increases computation |
| Any fixed $B$ | Deterministic under fixed model behavior and tie-breaking; no sampling is introduced |

A larger beam is not a higher-randomness setting. Returning one best sequence also does not measure output diversity. Better model probability need not correspond to better fluency, and wider search is not a guarantee of better text.

## Reproducibility checks

The original write-up showed different continuations for top-k with $k=1$ and beam width $B=1$. Under matching settings, those methods should make the same greedy choices. The archived outputs alone do not identify the cause of the mismatch.

Before using those examples as a comparison, rerun both paths with:

- The same checkpoint, vocabulary, tokenization, prompt, and initial recurrent state.
- Evaluation mode and consistent handling of dropout, special tokens, and stopping.
- Prompt tokens consumed exactly once and the correct state attached to each prefix.
- The same tie-breaking rule; fixed seeds for any genuinely stochastic decoder.

The repository entry point is `decoder.py`. Its comments record a historical environment with `torchtext==0.6.0` and `torch==1.13.1`; these are reproduction details, not a recommendation to install them into a current environment. The revised helper above does not replace the external repository's decoder, and generation has not been rerun for this review.

## Takeaways

- Sampling changes the distribution from which a token is drawn; beam search changes which prefixes survive a search.
- Nucleus filtering must keep the threshold-crossing token and renormalize.
- Recurrent-state and prompt-handling mistakes can invalidate an otherwise reasonable decoder comparison.
- Compare quality across multiple prompts and runs before concluding that one configuration is best.

## Sources

- [Original decoder and language model](https://github.com/kapshaul/NLP-sampling.search/tree/main)
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), which introduces nucleus sampling
