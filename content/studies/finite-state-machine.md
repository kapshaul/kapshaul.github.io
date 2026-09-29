---
title: "LSTMs: Parity, State Machines & POS Tagging"
date: 2024-05-10
lastmod: 2026-09-29
category: "NLP"
tags: ["LSTM", "Sequence Modeling", "POS Tagging"]
author: ["Yong-Hwan Lee"]
summary: "Explore what a small LSTM can remember through parity and grammar tasks, then examine a BiLSTM tagger's results and failure cases."
editPost:
    URL: "https://github.com/kapshaul/nlp-rnn-state-machines/tree/main"
    Text: "GitHub"
---

## Overview

This Oregon State University study examines recurrent models at three scales: a hand-configured scalar LSTM, learned finite-state behavior, and token-level part-of-speech (POS) tagging. The central question is whether a model learns a reusable state transition or only fits the sequences it sees during training.

The figures and reported training metrics below are from the original experiments. The explanations and example code have been reviewed against the repository; the revised code has not been used to retrain those models.

## A scalar LSTM for parity

Binary parity is a two-state problem. Reading a zero preserves the current state; reading a one flips it. With initial states $h_0=c_0=0$, the manually chosen gates are

$$
\begin{aligned}
i_t &= \sigma(10x_t+10h_{t-1}-5),\\
f_t &= \sigma(-10),\\
o_t &= \sigma(-10x_t-10h_{t-1}+15),\\
g_t &= \tanh(10),\\
c_t &= f_t c_{t-1}+i_t g_t,\\
h_t &= o_t\tanh(c_t).
\end{aligned}
$$

Classify the sequence as odd when $h_t\geq 0.5$, and even otherwise.

For Boolean-like inputs, the input gate approximates OR and the output gate approximates NAND. Since $f_t$ is near zero and $g_t$ is near one, the cell state approximately follows the input gate. **The cell state is not an AND gate.** The combination of the input and output gates produces XOR-like behavior: OR is active for either input, while NAND suppresses the output when both are active.

These are smooth gates, not exact Boolean operators. In particular, the positive hidden state is below one, and recurrence changes the gate inputs. The original `univariate_tester.py` checks all $2^{14}=16{,}384$ binary strings of length 14. Passing this finite test is evidence for the construction, not a proof for arbitrary sequence length.

## Learning and generalization

The learned parity model packs variable-length sequences, passes them through an LSTM, and classifies the final hidden state. The source driver's default training set contains every binary string of lengths 1–5. Evaluation samples 500 strings at each length from 1 to 256.

<div class="study-figure-grid">
<figure><img src="/finite-state-machine/LSTM-1_parity_generalization.png" alt="Parity accuracy by sequence length for hidden size 1"><figcaption>Hidden size 1</figcaption></figure>
<figure><img src="/finite-state-machine/LSTM-16_parity_generalization.png" alt="Parity accuracy by sequence length for hidden size 16"><figcaption>Hidden size 16</figcaption></figure>
<figure><img src="/finite-state-machine/LSTM-256_parity_generalization.png" alt="Parity accuracy by sequence length for hidden size 256"><figcaption>Hidden size 256</figcaption></figure>
</div>

The hidden-size-1 plot records perfect accuracy on the sampled evaluation sequences, including lengths much longer than the training examples. The other plots show that adding capacity does not automatically improve length generalization. These curves do not establish convergence speed or isolate the cause of errors; that would require training curves and repeated runs with controlled settings.

## Embedded Reber grammar

The embedded Reber grammar shown below is a **regular language represented by a finite-state machine**. Its internal loops can produce long strings, but it does not require an unbounded stack or recursive nesting.

An outer branch chooses `T` or `P`, traverses an inner Reber grammar, and requires the matching branch symbol before the final `E`. For example, a valid path in this diagram is:

```text
B T [B T X S E] T E
```

The brackets explain the inner grammar; they are not part of the generated string. The resulting string is `BTBTXSETE`. Remembering the outer branch while traversing the inner graph creates a delayed dependency.

<figure>
<img src="/finite-state-machine/erg.png" alt="Embedded Reber grammar with matching outer T and P branches">
<figcaption>Embedded Reber grammar used in the original study.</figcaption>
</figure>

<figure>
<img src="/finite-state-machine/graph.png" alt="Archived training and validation accuracy comparison for RNN and LSTM">
<figcaption>Original RNN/LSTM comparison: both fit the training set, while the LSTM has higher validation accuracy in this recorded run.</figcaption>
</figure>

The comparison is consistent with gated memory helping on this task. It does not prove that LSTMs always generalize better, or that their gradients never vanish. The linked repository's parity and POS drivers do not provide enough information to reproduce this archived grammar comparison independently.

## POS tagging with a BiLSTM

The POS experiment uses the English UDPOS dataset. A bidirectional LSTM combines left and right context before assigning a tag to each token. This is appropriate when the whole sentence is available.

<figure>
<img src="/finite-state-machine/Histogram.png" alt="Frequency of part-of-speech labels in the dataset">
<figcaption>Original label distribution. Class imbalance makes a majority-label baseline a useful comparison.</figcaption>
</figure>

### Correctly excluding padding

The original driver pads target labels with `0` and uses `CrossEntropyLoss()` without an ignore index for those padded targets. Consequently, padded positions contribute to training loss even though test accuracy is computed over real tokens. This should be corrected before rerunning the experiment.

The following example uses a distinct target padding value, `-100`. Input padding IDs and target padding values serve different purposes. It returns raw logits because cross-entropy applies log-softmax internally.

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tag_size, pad_id, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, bidirectional=True, batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(2 * hidden_dim, tag_size)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=tokens.size(1)
        )
        return self.classifier(self.dropout(output))


def tagging_loss(logits, targets):
    # targets: [batch, sequence], with -100 at every padded position
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
    )
```

### Recorded performance

<figure>
<img src="/finite-state-machine/Loss.png" alt="Training and validation loss across POS tagging epochs">
<figcaption>Original loss curves. Compare training loss with validation loss; accuracy and loss are different quantities.</figcaption>
</figure>

| Original report, epoch 40 | Value |
| --- | ---: |
| Training loss | 0.0227 |
| Validation loss | 0.2679 |
| Test token accuracy | 86.23% |

These are historical results from the original implementation, including its padding behavior. They are not results of the corrected example above. The training–validation gap warrants checking overfitting and the loss calculation; the plot alone cannot attribute the gap to unknown tokens or prove that dropout fixed it.

### Error analysis

The original inference examples contain errors and should not be presented as correct reference tags. These garden-path sentences require a reading that may differ from a token's most frequent use.

| Sentence | Token | Recorded prediction | Intended tag in this reading |
| --- | --- | --- | --- |
| The old man the boat. | man | NOUN | VERB: the old people operate the boat |
| The complex houses married and single soldiers and their families. | houses | NOUN | VERB: the complex accommodates people |
| The man who hunts ducks out on weekends. | hunts | PROPN | VERB: “who hunts” modifies “the man” |

In the third sentence, `ducks` is the main verb in “ducks out,” and the recorded model already tags it as VERB. These examples illustrate contextual ambiguity; they do not measure performance across the test set.

## Reproduction notes

| Entry point | Purpose |
| --- | --- |
| `univariate_tester.py` | Exhaustive length-14 test of the manually configured cell |
| `driver_parity.py` | Train parity models and evaluate length generalization |
| `driver_udpos.py` | Train and evaluate the POS tagger |

The repository uses historical TorchText APIs. Reproduction requires a compatible Python/PyTorch/TorchText environment and the original data, rather than an unpinned install of the newest packages. Before comparing new results, record versions and seeds, correct target padding, and report accuracy only on non-padding tokens.

## Takeaways

- A compact recurrent state can represent parity; sampled or bounded tests still have a limited scope.
- Embedded Reber grammar tests delayed finite-state dependencies, not unrestricted recursion.
- Padding and loss definitions are part of the experiment. They must agree with the evaluation mask.
- Inspecting ambiguous sentences reveals errors that a single accuracy score hides.

## Sources

- [Original experiment repository](https://github.com/kapshaul/nlp-rnn-state-machines/tree/main)
- [PyTorch LSTM documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch cross-entropy documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
