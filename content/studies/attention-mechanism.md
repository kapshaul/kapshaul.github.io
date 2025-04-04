---
title: "Natural Language Processing (NLP) - Attention Mechanisms in Sequence-to-Sequence Models" 
date: 2024-06-07
lastmod: 2024-07-12
tags: ["NLP","Machine Learning","Deep Learning","Attention Mechanism","Transformer","BLEU Score","ArtificialIntelligence"]
author: ["Yong-Hwan Lee"]
description: "This study was carried out as a project at Oregon State University."
summary: "Explore the attention mechanism's role in enhancing context and accuracy in machine translation, with quality assessed by the BLEU score."
editPost:
    URL: "https://github.com/kapshaul/NLP-attention.mechanism"
    Text: "GitHub"
showToc: true
disableAnchoredHeadings: false

---

## Overview

The main objective of this project is to understand scaled dot-product attention and to implement a simple attention mechanism in a sequence-to-sequence model. The task involves translating sentences from German to English using the Multi30k dataset, which contains over 31,000 bitext sentences describing common visual scenes in both languages.

---

## Scaled Dot-Product Attention

Starts from the definition of a single-query scaled dot-product attention mechanism. Given a query $\mathbf{q} \in \mathbb{R}^{1\times d}$, a set of candidates represented by keys $\mathbf{k}_1, ... , \mathbf{k}_m \in \mathbb{R}^{1\times d}$ and values $\mathbf{v}_1, ... , \mathbf{v}_m \in \mathbb{R}^{1\times d_v}$, we compute the scaled dot-product attention as:

$$
\alpha_i = \frac{\text{exp}\left(~\mathbf{q}\mathbf{k}_ i^T / \sqrt d\right)}{\sum_{j=1}^m \text{exp}\left(\mathbf{q}\mathbf{k}_ j^T / \sqrt d\right)}
$$

$$
\textbf{a} = \sum_{j=1}^m \alpha_j \mathbf{v}_j
$$

where the $\alpha_i$ are referred to as attention values (or collectively as an attention distribution).

#### **1.1. Copying**
Q. Describe what properties of the keys and queries would result in the output $\textbf{a}$ being equal to one of the input values $\mathbf{v}_j$. Specifically, what must be true about the query $\mathbf{q}$ and the keys $\mathbf{k}_1, ..., \mathbf{k}_m$ such that $\textbf{a} \approx \mathbf{v}_j$? (We assume all values are unique -- $\mathbf{v}_i \neq \mathbf{v}_j,~\forall ij$.)

> In the case where $\textbf{a} \approx \mathbf{v}_j$, the similarity score for $\mathbf{q}\mathbf{k}_j^T$ is significantly higher than all others due to the softmax function producing outputs to have probability distribution. Therefore, given a query $\mathbf{q}$, $\mathbf{k}_j$ must be significantly higher than others to determine the similarity score.

#### **1.2. Average of Two**
Q. Consider a set of key vectors $\mathbf{k}_1, ... , \mathbf{k}_m$ where all keys are orthogonal unit vectors -- that is to say $\mathbf{k}_i \mathbf{k}_j^T = 0, \forall ij$ and $\Vert\mathbf{k}_i\Vert=1,\forall i$. Let $\mathbf{v}_a, \mathbf{v}_b \in \{\mathbf{v}_1, ..., \mathbf{v}_m\}$ be two value vectors. Give an expression for a query vector $\mathbf{q}$ such that the output $\textbf{a}$ is approximately equal to the average of $\mathbf{v}_a$ and $\mathbf{v}_b$, that is to say $\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)$. You can reference the key vectors corresponding to $\mathbf{v}_a$ and $\mathbf{v}_b$ as $\mathbf{k}_a$ and $\mathbf{k_b}$ respectively.

>From $\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)$, we can consider the term $\frac{1}{2}$ is from $\alpha_i$. Meaning that $\alpha_a = \alpha_b$ and $\alpha_i = 0$ should be satisfied to meet the condition. Since $\alpha_i = \text{softmax}(\mathbf{q}\mathbf{k}_i^T)$, we only want to keep $\mathbf{k}_a$ and $\mathbf{k}_b$; otherwise, $\mathbf{k}_i=0$.
>
>By constructing $\mathbf{q}=\mathbf{k}_a + \mathbf{k}_b$, we can ensure if this expression satisfy the condition.
>
>$$
\mathbf{q}\mathbf{k}_a^T=(\mathbf{k}_a\mathbf{k}_a^T + \mathbf{k}_b\mathbf{k}_a^T)=(1+0)=1
$$
>
>$$
\mathbf{q}\mathbf{k}_b^T=(\mathbf{k}_a\mathbf{k}_b^T + \mathbf{k}_b\mathbf{k}_b^T)=(0+1)=1
$$
>
>$$
\mathbf{q}\mathbf{k}_i^T=(\mathbf{k}_a\mathbf{k}_i^T + \mathbf{k}_b\mathbf{k}_i^T)=(0+0)=0
$$
>
>After applying the softmax function, given that $\mathbf{q}\mathbf{k}_a^T = 1$, $\mathbf{q}\mathbf{k}_b^T = 1$, and $\mathbf{q}\mathbf{k}_i^T = 0$, the resulting attention weights are approximately $\alpha_a \approx \frac{1}{2}$ and $\alpha_b \approx \frac{1}{2}$.
>Therefore $\textbf{a}$ can be written,
>
>$$
\textbf{a} \approx \frac{1}{2}\mathbf{v}_a + \frac{1}{2}\mathbf{v}_b + 0
$$
>
>$$
\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)
$$

#### **1.3. Noisy Average**
Q. Now consider a set of key vectors $\{\mathbf{k}_1, ... , \mathbf{k}_m\}$ where keys are randomly scaled such that $\mathbf{k}_i = \mathbf{\mu}_i*\lambda_i$ where $\lambda_i \sim \mathcal{N}(1, \beta)$ is a randomly sampled scalar multiplier. Assume the unscaled vectors $\mu_1, ..., \mu_m$ are orthogonal unit vectors. If you use the same strategy to construct the query $q$ as you did in Task 1.2, what would be the outcome here? Specifically, derive $\mathbf{q}\mathbf{k}_a^T$ and $\mathbf{q}\mathbf{k}_b^T$ in terms of $\mu$'s and $\lambda$'s. Qualitatively describe what how the output $a$ would vary over multiple resamplings of $\lambda_1, ..., \lambda_m$.

>From the expression for $\mathbf{q}$ in Task 1.2,
>
>$$
\mathbf{q}=\mathbf{k}_a + \mathbf{k}_b
$$
>
>By substituting $\mathbf{k}_i = \mathbf{\mu}_i*\lambda_i$,
>
>$$
\mathbf{q}=\mathbf{\mu}_a*\lambda_a + \mathbf{\mu}_b*\lambda_b
$$
>
>The expression for $\mathbf{q}\mathbf{k}_a^T$ and $\mathbf{q}\mathbf{k}_b^T$,
>
>$$
\mathbf{q}\mathbf{k}_a^T=\lambda_a^2*\mathbf{\mu}_a\mathbf{\mu}_a^T + \lambda_a\lambda_b*\mathbf{\mu}_b\mathbf{\mu}_a^T)=\lambda_a^2
$$
>
>$$
\mathbf{q}\mathbf{k}_b^T=(\lambda_a\lambda_b*\mathbf{\mu}_a\mathbf{\mu}_b^T + \lambda_b^2*\mathbf{\mu}_b\mathbf{\mu}_b^T)=\lambda_b^2
$$
>
>Now, we can consider 3 different cases,
>
>Case 1. $\lambda_a \approx \lambda_b$,
>
>$$
\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)
$$
>
>Case 2. $\lambda_a \gg \lambda_b$,
>
>$$
\textbf{a} \approx \mathbf{v}_a
$$
>
>Case 3. $\lambda_a \ll \lambda_b$,
>
>$$
\textbf{a} \approx \mathbf{v}_b
$$
>
>Since randomly sampled following $\lambda_i \sim \mathcal{N}(1, \beta)$,
>
>$$
\mathbb{E}[\mathbf{q}\mathbf{k}_a^T]=\mathbb{E}[\lambda_a^2]=1
$$
>
>$$
\mathbb{E}[\mathbf{q}\mathbf{k}_b^T]=\mathbb{E}[\lambda_b^2]=1
$$
>
>Over multiple resamplings of $\lambda_1, ..., \lambda_m$,
>
>$$
\mathbb{E}[\textbf{a}] = \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)
$$

#### **1.4. Noisy Average with Multi-head Attention**
Q. Let's now consider a simple version of multi-head attention that averages the attended features resulting from two different queries. Here, two queries are defined ($\mathbf{q}_1$ and $\mathbf{q}_2$) leading to two different attended features ($\textbf{a}_1$ and $\textbf{a}_2$). The output of this computation will be $\textbf{a} = \frac{1}{2}(\textbf{a}_1 + \textbf{a}_2)$. Assume we have keys like those in Task 1.3, design queries $\mathbf{q}_1$ and $\mathbf{q}_2$ such that $\textbf{a} \approx \frac{1}{2}(\mathbf{v}_a + \mathbf{v}_b)$.

>A simple strategy is to design each query so that one head selects $q_1=\mathbf{v}_a$ exclusively and the other head selects $q_2=\mathbf{v}_b$ exclusively. Then, by averaging those two attended features,
>
>$$
\mathbf{q}_1=\lambda_a*\mathbf{\mu}_a
$$
>
>$$
\mathbf{q}_2=\lambda_b*\mathbf{\mu}_b
$$
>
>By constructing the linear equation of $\mathbf{q}$ and $\mathbf{k}$,
>
>$$
\mathbf{q}_1\mathbf{k}_a^T=\lambda_a*\mathbf{\mu}_a\mathbf{k}_a^T =\lambda_a^2*(\mathbf{\mu}_a\mathbf{\mu}_a^T)=\lambda_a^2
$$
>
>$$
\mathbf{q}_2\mathbf{k}_b^T=\lambda_b*\mathbf{\mu}_b\mathbf{k}_b^T =\lambda_b^2*(\mathbf{\mu}_b\mathbf{\mu}_b^T)=\lambda_b^2
$$
>
>From here, $\textbf{a}_1$ and $\textbf{a}_2$ can be expressed,
>
>$$
\alpha_i=\text{softmax}(\mathbf{q}_1\mathbf{k}_i^T)
\hspace{20pt}
\alpha_j=\text{softmax}(\mathbf{q}_2\mathbf{k}_j^T)
$$
>
>$$
\alpha_i = 
\begin{cases} 
1 & \text{if } i = a \\
0 & \text{if } i \neq a 
\end{cases}
\hspace{20pt}
\alpha_j = 
\begin{cases} 
1 & \text{if } j = b \\
0 & \text{if } j \neq b
\end{cases}
$$
>
>$$
\textbf{a}_1 = \alpha_a \mathbf{v}_a = \mathbf{v}_a
\hspace{40pt}
\textbf{a}_2 = \alpha_b \mathbf{v}_b = \mathbf{v}_b
$$
>
>The final output is the average of $\textbf{a}_1$ and $\textbf{a}_2$.
>
>$$
\textbf{a} = \frac{1}{2}(\textbf{a}_1 + \textbf{a}_2)
$$

---

## German-to-English Machine Translation

Machine translation involves automatically converting text from one language to another using computational models. In the context of translating German to English, Scaled-Dot Product Attention is a core mechanism used in modern neural networks, particularly in models like the **Transformer**. This mechanism allows the model to focus on relevant parts of the input sentence when generating the output translation, enabling it to capture the nuances and structure of German and translate them effectively into English.

By using Scaled-Dot Product Attention, the model effectively learns these common patterns and differences between German and English

<br>

#### **2.1. German-to-English Attention Pattern**

<div align="center">

**Example 1**
<img src="/attention-mechanism/104_translation.png" width="300">
**Example 2**
<img src="/attention-mechanism/114_translation.png" width="300">

**Example 3**
<img src="/attention-mechanism/281_translation.png" width="300">
**Example 4**
<img src="/attention-mechanism/759_translation.png" width="300">

**Figure 1**: Attention diagram examples

</div>

- **SVO vs SOV**

English typically follows an SVO (Subject-Verb-Object) structure, while German often employs an SOV (Subject-Object-Verb) structure in subordinate clauses. This difference leads to deviations from the diagonal in the verb area on attention maps, as illustrated in Examples 1 and 2. For instance, deviations can be seen in the phrases "are climbing" in Example 1 and "are anticipating" in Example 2.

- **Preposition and Conjunction**

Prepositions and their objects usually align closely, but their usage can differ in certain cases. For example, in Example 3, the English preposition "to" corresponds to the German preposition "zu," and the English conjunction "while" corresponds to the German conjunction "während." Both of these cases show deviations rather than diagonal alignment. In Example 4, the English preposition "in front of" corresponds to the German preposition "vor," which also shows significant deviation.

<br>

#### **2.2. Scaled-Dot Product Attention Code**

```python
class SingleQueryScaledDotProductAttention(nn.Module):    
    # kq_dim is the dimension of keys and queries. Linear layers should be used to project inputs to these dimensions.
    def __init__(self, enc_hid_dim, dec_hid_dim, kq_dim=512):
        super().__init__()
        self.W_k = nn.Linear(enc_hid_dim * 2, kq_dim)
        self.W_q = nn.Linear(dec_hid_dim, kq_dim)
        self.kq_dim = kq_dim

        
    #hidden is h_t^{d} from Eq. (11) and has dim => [batch_size, dec_hid_dim]
    #encoder_outputs is the word representations from Eq. (6) 
    # and has dim => [src_len, batch_size, enc_hid_dim * 2]
    def forward(self, hidden, encoder_outputs):
        # Compute for q = W_q h_j^{e}
        query = self.W_q(hidden).unsqueeze(1)
        # Compute for k_t = W_k h_t^{d}
        key = self.W_k(encoder_outputs)
        # Compute for score = qk^T / sqrt(d)
        score = torch.bmm(query, key.T(1, 2)).squeeze(1) / np.sqrt(self.kq_dim)
        # Compute alpha = softmax(score)
        alpha = F.softmax(score, dim=1)
        # Compute a = sum(alpha_j v_j)
        attended_val = torch.sum(encoder_outputs * alpha.unsqueeze(2), dim=1)
        
        assert attended_val.shape == (hidden.shape[0], encoder_outputs.shape[2])
        assert alpha.shape == (hidden.shape[0], encoder_outputs.shape[0])
        return attended_val, alpha
```

<br>

#### **2.3. BLEU Score Comparision**

Models were trained and evaluated using the *Dummy baseline*, *MeanPool*, and *attention mechanisms*. For each method, three independent runs were conducted, and the mean and variance of the results were calculated to compare performance and consistency.

<div align="center">
  
| Attn. | PPL Mean | PPL Var | BLEU Mean | BLEU Var |
|:-----:|:--------:|:-------:|:---------:|:--------:|
| None  |  18.107  |  0.057  |   16.3    |  0.118   |
| Mean  |  15.809  |  0.028  |  18.599   |  0.277   |
| SDP   |  10.447  |  0.049  |  34.64    |  0.793   |

</div>

The trends indicate that the scaled dot-product attention mechanism (SDP), significantly improve the model's performance, as lower perplexity and higher BLEU scores. However, the increased variance in BLEU scores with SDP suggests that there might be some variability caused by random initialization.
