---
title: "Natural Language Processing (NLP) - Sampling Search "
date: 2024-03-12
lastmod: 2024-03-12
tags: ["NLP","LSTM","Sampling Search","Text Generation","Beam Search","Temperature Scaled","Top-k","Top-p"]
author: ["Yong-Hwan Lee"]
description: "This study was carried out as a project at Oregon State University."
summary: "Explore decoding methods for language models by implementing sampling-based (vanilla, temperature-scaled, top-k, top-p) and search-based (beam search) techniques using a 3-layer LSTM trained on Game of Thrones text, and analyze their text generation behavior."
editPost:
    URL: "https://github.com/kapshaul/NLP-sampling.search"
    Text: "GitHub"
showToc: true
disableAnchoredHeadings: false

---

## Overview

This study explores a **pre-trained language model** built with a multi-layer LSTM architecture. The model is trained on text from the first five *Game of Thrones* novels, capturing both short- and long-range dependencies in the data. By examining its internal architecture and forward pass, we gain insight into how it transforms input tokens into meaningful hidden representations, ultimately predicting the next token in a sequence.

Key points include:
- **3-layer LSTM** with an embedding layer for token representations.
- **Hidden size of 512**, capturing rich contextual information.
- **Vocabulary management** using a separate text processing pipeline.
- **Trained on a large fantasy corpus**, showcasing the model’s capacity to learn diverse linguistic patterns.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/kapshaul/NLP-sampling.search
cd NLP-sampling.search
pip install torchtext==0.6.0 torch==1.13.1
```

---

## Implementation

To test search configurations (temperature, top-k, top-p, beam search), run:

```bash
python decoder.py
```

---

## Sampling-based Decoding

This section explores various stochastic decoding strategies for autoregressive language generation using the pre-trained LSTM model. The goal is to sample text sequences from the model under different probabilistic constraints, which affect creativity, coherence, and diversity in generated text.

#### Decoding Methods Implemented

1. **Vanilla Sampling**  
   At each step, the next token is sampled directly from the softmax distribution of the model’s output logits. This approach retains the full probability distribution, offering high variance in the results.

2. **Temperature-Scaled Sampling**  
   A temperature parameter $\tau$ is introduced to control the sharpness of the softmax distribution:
   - $\tau < 1$: Sharper distributions; more deterministic behavior.
   - $\tau > 1$: Flatter distributions; increased randomness.
   - $\tau = 1$: Equivalent to vanilla sampling.

3. **Top-k Sampling**  
   Restricts the sampling pool to the top $k$ most probable tokens, setting all others to zero before re-normalization. This limits randomness to a focused subset of likely candidates.

4. **Nucleus (Top-p) Sampling**  
   Selects the smallest possible set of words whose cumulative probability exceeds $p$. This dynamically adjusts the candidate set size based on distribution shape, balancing control and diversity.

#### Testing Overview

Each of these strategies was implemented within a unified `sample()` function that supports prompt conditioning and customizable parameters (`temp`, `k`, `p`). A single model forward pass is performed at each step, and the output distribution is adjusted based on the chosen sampling strategy.

```python
def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1):
    assert (k == 0 or p == 1), "Cannot combine top-k and top-p sampling"
    ...
    return decodedString
```

#### Prompt Conditioning

Sampling can be initialized with a text prompt. The prompt is numeralized and passed through the model to update the internal hidden states before generation begins.

#### Visualization of Sampling Behavior

The following table summarizes outputs generated from the prompt:

**Prompt**: `"the night is dark and full of terrors"`

| Method                   | Settings               | Notable Behavior                                                  |
|--------------------------|------------------------|--------------------------------------------------------------------|
| Vanilla Sampling         | temp = 1               | High variance; coherent but unpredictable stories                 |
| Temperature-scaled       | τ = 0.0001             | Very deterministic, repetitive or generic continuations           |
| Temperature-scaled       | τ = 100                | Extreme randomness; nonsensical token-level outputs               |
| Top-k Sampling           | k = 1                  | Very deterministic (equivalent to greedy search)                  |
| Top-k Sampling           | k = 20                 | Balanced between diversity and fluency                            |
| Top-p Sampling           | p = 0.001              | Similar to top-1, often repetitive                                 |
| Top-p Sampling           | p = 0.75               | Naturally diverse yet still contextually reasonable               |
| Top-p Sampling           | p = 1                  | Equivalent to vanilla sampling                                    |

#### Example Output (Top-p, p=0.75)

> *"the night is dark and full of terrors . with the ryswells , the knights of the golden mountains burst off and come down in the attempt..."*

This illustrates the potential for coherent storytelling using nucleus sampling while avoiding overly deterministic sequences.

#### Observations

- Lower temperatures and small top-k or top-p values produce more deterministic results.
- Higher temperatures and larger values introduce randomness and narrative exploration.
- Nucleus sampling (`top-p`) offers an adaptive alternative to fixed cutoffs, providing smoother trade-offs between creativity and coherence.

---

## Search-based Decoding with Beam Search

Unlike stochastic sampling methods, **beam search** is a deterministic decoding strategy that aims to identify the most probable sequence under the model. It maintains multiple hypotheses at each time step, expanding and retaining only the top candidates based on cumulative probability.

### Beam Search Algorithm

At each time step `t`, beam search performs two operations:

1. **Expansion**: Each current hypothesis (beam) is extended by all possible next words from the vocabulary.
2. **Selection**: The resulting candidates are scored using their cumulative log-probabilities. Only the top `B` beams are kept for the next step.

Formally, the score of a candidate is computed as:

```
logP(w₀, ..., wₜ, w) = logP(w₀, ..., wₜ) + logP(w | w₀, ..., wₜ)
```

This process is repeated until a specified maximum sequence length is reached. The candidate with the highest score is returned as the final output.

### Testing Overview

The `beamsearch()` function is implemented with support for prompt conditioning and variable beam width:

```python
def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
    ...
    return decodedString
```

Key features:
- Maintains `B` candidate sequences at each step.
- Tracks hidden and cell states for each beam.
- Performs efficient batched inference using PyTorch.

### Beam Width Comparison

Using the same prompt `"the night is dark and full of terrors"`, the model is evaluated under different beam widths:

| Beam Width | Sample Output Snippet |
|------------|------------------------|
| B = 1      | "a smile , the storm girl staring , and the shapes were beautiful..." |
| B = 10     | "meereen is deceit , and the common soldiers held through the woods..." |
| B = 50     | "meereen was laid up , crowned with three - finger hobb with a pair of faces..." |

### Observations

- **B = 1** acts like greedy decoding — fast but potentially shortsighted.
- **B = 10** strikes a balance between coherence and diversity.
- **B = 50** provides highly diverse outputs but may generate less coherent or overly ornate sequences.

While beam search increases decoding stability and reduces randomness, larger beam sizes also increase computation and don't always guarantee better fluency. It is important to balance performance and quality depending on application needs.

---

## Example Outputs

Below are selected examples from different decoding strategies applied to the prompt:

**Prompt**: `"the night is dark and full of terrors"`

#### Sampling-based Decoding

- **Vanilla Sampling**  
  > "the night is dark and full of terrors . after no one was dead . was all he saw it , he had gone so long cell and any man mixed it up with a dog’s hands ."

- **Temperature-scaled Sampling (τ = 0.0001)**  
  > "the night is dark and full of terrors . with stannis and most of the queen’s men gone , her flock was much diminished; half a hundred of the free folk to defend the vale..."

- **Temperature-scaled Sampling (τ = 100)**  
  > "the night is dark and full of terrors herring depart: endearments cargoes tucked areo confessed frost traces prepared piety crude fortune nowhere miss betoken whistles..."

- **Top-k Sampling (k = 1)**  
  > "the night is dark and full of terrors . with stannis and most of the queen’s men gone , her flock was much diminished..."

- **Top-k Sampling (k = 20)**  
  > "the night is dark and full of terrors . though tyrion had the sort of <unk> being returned to the new . she had forgotten who she was..."

- **Top-p Sampling (p = 0.001)**  
  > "the night is dark and full of terrors . with stannis and most of the queen’s men gone , her flock was much diminished..."

- **Top-p Sampling (p = 0.75)**  
  > "the night is dark and full of terrors . with the ryswells , the knights of the golden mountains burst off and come down in the attempt..."

- **Top-p Sampling (p = 1)**  
  > "the night is dark and full of terrors . after no one was dead . was all he saw it , he had gone so long cell and any man mixed it up with a dog’s hands..."

#### Search-based Decoding (Beam Search)

- **Beam Search (B = 1)**  
  > "the night is dark and full of terrors . a smile , the storm girl staring , and the shapes were beautiful . all the vaults are rising from horizon were wind and branches..."

- **Beam Search (B = 10)**  
  > "the night is dark and full of terrors . meereen is deceit , and the common soldiers held through the woods and down the waters they could walk..."

- **Beam Search (B = 50)**  
  > "the night is dark and full of terrors . meereen was laid up , crowned with three - finger hobb with a pair of faces beneath the silk - and - white banner..."
