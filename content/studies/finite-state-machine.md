---
title: "Natural Language Processing (NLP) - Finite State Machine with RNN"
date: 2024-05-10
lastmod: 2024-05-10
tags: ["NLP","LSTM","RNN","Sequence Modeling","Machine Learning"]
author: ["Yong-Hwan Lee"]
description: "This study was carried out as a project at Oregon State University."
summary: "This study focuses on using LSTMs for parity detection, finite state machine learning, and part-of-speech tagging with a BiLSTM. It applies theoretical concepts and PyTorch to build and evaluate sequential models."
editPost:
    URL: "https://github.com/kapshaul/NLP-finite.state.machine.RNN"
    Text: "GitHub"
showToc: true
disableAnchoredHeadings: false

---

## Overview

This study demonstrates the power of recurrent neural networks (RNNs), particularly long short-term memory (LSTM) models, across a range of natural language processing tasks. It begins with a manually engineered LSTM for binary parity classification and progresses to training LSTM networks for generalization, embedded Reber grammar recognition, and part-of-speech tagging using a BiLSTM.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/kapshaul/NLP-finite.state.machine.RNN
cd NLP-finite.state.machine.RNN
pip install torch torchtext matplotlib portalocker
```

---

## Implementation

1. To test **manual LSTM configuration for binary parity**, run:

   ```bash
   python univariate_tester.py
   ```

2. To train and evaluate **LSTM models on parity tasks**, run:

   ```bash
   python driver_parity.py
   ```

3. To implement and train **BiLSTM for POS tagging**, run:

   ```bash
   python driver_udpos.py
   ```

---

## Demystifying Recurrent Neural Networks

#### Hand Designing an LSTM for Parity

In this section, we manually explore the capability of LSTM networks to handle sequential tasks, specifically determining the parity of binary strings (i.e., whether the number of ones is even or odd). A simple binary string parity classification can be represented by recursive XOR operations, an ideal use-case for an LSTM's recurrent structure.

#### Univariate LSTM Setup
We can consider a LSTM where inputs, outputs, and weights are scalars, defined by:

$i_t = \sigma(w_{ix}x_t + w_{ih}h_{t-1} + b_i)$<br>
$f_t = \sigma(w_{fx}x_t + w_{fh}h_{t-1} + b_f)$<br>
$o_t = \sigma(w_{ox}x_t + w_{oh}h_{t-1} + b_o)$<br>
$g_t = \tanh(w_{gx}x_t + w_{gh}h_{t-1} + b_g)$<br>
$c_t = f_t c_{t-1} + i_t g_t$<br>
$h_t = o_t \tanh(c_t)$

#### Manual Parameter Setting for XOR
We can find weights and biases to perform parity classification manually. The goal was to have the final hidden state (`h_t`) ≥ 0.5 for odd parity and < 0.5 for even parity. The selected weights and biases:

- **Input gate**: `wix = 10`, `wih = 10`, `bi = -5`
- **Forget gate**: `wfx = 0`, `wfh = 0`, `bf = -10`
- **Output gate**: `wox = -10`, `woh = -10`, `bo = 15`
- **Gate `g`**: `wgx = 0`, `wgh = 0`, `bg = 10`

With these parameters:
- `i_t` acts as an OR gate.
- `o_t` acts as a NAND gate.
- `c_t` effectively behaves as an AND gate.

This demonstrates that even a minimal LSTM can solve the parity problem through careful manual configuration.

#### Understanding
We have demonstrated that a single-dimensional LSTM can theoretically compute parity for binary sequences of arbitrary length, setting the foundation to later explore learning these parameters automatically.


---

## Learning Finite State Machines with LSTM

This section explores the capacity of LSTMs to model deterministic finite state machines (FSMs), including both synthetic binary classification and structured language sequences.

#### Parity Task: Generalization to Longer Sequences

The LSTM is trained on binary sequences of varying lengths to predict their parity (even or odd number of 1s). The dataset is generated with all binary combinations up to a `max_length`, and the labels are calculated as `sum(seq) % 2`.

#### Model Summary

```python
class ParityLSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, lengths):
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.rnn(x)
        output = self.fc(h_n[-1])
        return output
```

- Input is padded to batch format, and packed before passing to the LSTM.
- Final hidden state is used for classification via a fully connected output layer.

#### Understanding

<div align="center">

<div style="display: flex; gap: 10px; justify-content: center; align-items: flex-start; flex-wrap: wrap;">

  <div style="text-align: center;">
    <img src="/finite-state-machine/LSTM-1_parity_generalization.png" width="230">
    <div>(a) Hidden size = 1</div>
  </div>

  <div style="text-align: center;">
    <img src="/finite-state-machine/LSTM-16_parity_generalization.png" width="230">
    <div>(b) Hidden size = 16</div>
  </div>

  <div style="text-align: center;">
    <img src="/finite-state-machine/LSTM-256_parity_generalization.png" width="230">
    <div>(c) Hidden size = 256</div>
  </div>

</div>

**Figure 1**: LSTM Parity Detection Accuracy with Varying Hidden Sizes

</div>

- LSTM with hidden size of **1** can learn the task to **100% accuracy**, validating the theoretical result.
- Larger hidden sizes speed up convergence but may **overfit** on shorter training sequences.
- Generalization to sequences up to **length 256** is tested, and performance drops slightly unless tuned properly.

---

#### Embedded Reber Grammar (ERG): Recognizing Structured Sequences

The LSTM is further challenged with a more complex task: classifying whether a string was generated by a structured **Embedded Reber Grammar (ERG)**.
To evaluate how recurrent models handle structured, long-range dependencies, we use the **Embedded Reber Grammar (ERG)** task. This synthetic task challenges a model to classify whether a given sequence follows the strict rules of ERG generation.

#### What is the Embedded Reber Grammar?

The **Embedded Reber Grammar** is a state machine used to generate sequences of characters following recursive, nested patterns. It contains two identical sub-networks (Reber grammars) that can repeat multiple times before the sequence ends.

- A valid ERG string example:  
  `BTBTXSEBTXSEBPVVEBTXXVVETE`

  This decomposes into:  
  `BT | BTXSE | BTXSE | BPVVE | BTXXVVE | TE`

- The task is to classify whether a given sequence is valid (follows ERG rules) or invalid (e.g., due to character-level perturbations).

<div align="center">
    
<img src="/finite-state-machine/erg.png" width="500">

**Figure 2**: ERG generation diagram

</div>

#### Models Compared

Two models are evaluated on this task:

| Model        | Train Accuracy | Validation Generalization |
|--------------|----------------|----------------------------|
| RNN          | High           | Poor (overfits)            |
| LSTM         | High           | Strong generalization      |

<div align="center">
    
<img src="/finite-state-machine/graph.png" width="500">

**Figure 3**: RNN vs LSTM

</div>

- **RNN**: Struggles with long-term dependencies, fails to generalize despite fitting the training set.
- **LSTM**: Learns the underlying recursive structure and performs well on unseen examples.

#### Why LSTM Outperforms RNN?

LSTM's design includes key architectural features:
- **Input, forget, and output gates** allow selective memory retention.
- **Cell state** enables long-distance signal propagation without degradation.
- **Effective for recursion and repeated structures**, unlike RNNs, which suffer from vanishing gradients.

As a result, LSTMs can maintain context across complex, nested subsequences — which is essential for modeling grammars like ERG.

---

## Part-of-Speech Tagging with BiLSTM

This task applies BiLSTM models to a real-world NLP application — tagging each word in an English sentence with its corresponding part-of-speech (POS) using the [UDPOS dataset](https://universaldependencies.org/).

#### Dataset Overview

- Comes with `train`, `valid`, and `test` splits.
- Includes a mix of topics (e.g., family, employment, science).
- POS distribution is **imbalanced**, so majority label baseline is used as a sanity check.

#### Preprocessing

- Custom `pad_collate()` is used to batch variable-length sequences.
- Lemmatization is not applied, but could help reduce sparsity.
- Words are converted to token IDs via a vocabulary object or `torchtext` pipeline.

<div align="center">
    
<img src="/finite-state-machine/Histogram.png" width="500">

**Figure 4**: POS Histogram

</div>

#### BiLSTM Model Architecture

```python
class BILSTM_POS(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, tag_size)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        o, _ = self.bilstm(x)
        o, _ = pad_packed_sequence(o, batch_first=True)
        o = self.dropout(o)
        o = self.fc(o)
        return torch.log_softmax(o, dim=-1)
```

- Embedding layer → BiLSTM → dropout → linear output → log-softmax over tags
- Bidirectional structure ensures each word is contextualized with both left and right neighbors.

#### Training Observations

- Training accuracy improves steadily and outpaces validation loss after ~30 epochs.
- Likely due to over-representation of common tokens like `UNK`, which default to the `NOUN` tag early in training.
- Dropout regularization helps mitigate overfitting.

<div align="center">
    
<img src="/finite-state-machine/Loss.png" width="500">

**Figure 5**: Train and Validation Loss

</div>

#### Loss Trend

```text
Epoch 40/40
Train Loss: 0.0227
Valid Loss: 0.2679
Test Accuracy: 86.23%
```

#### POS Tagging Inference Examples

**Example 1:**  
`The old man the boat.`  
`DET ADJ NOUN DET NOUN PUNCT`

**Example 2:**  
`The complex houses married and single soldiers and their families.`  
`DET ADJ NOUN VERB CCONJ ADJ NOUN CCONJ PRON NOUN PUNCT`

**Example 3:**  
`The man who hunts ducks out on weekends.`  
`DET NOUN PRON PROPN VERB ADV ADP NOUN PUNCT`
