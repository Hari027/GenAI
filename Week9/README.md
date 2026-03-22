# CSET419 – Generative AI | Lab 9
### Generative Models for Sequential Data

---

## Overview

This lab explores how generative models learn patterns from sequential (text) data and produce new sequences. Two architectures are implemented and compared side by side:

- **Component I** — LSTM-based Language Model
- **Component II** — Transformer-based Language Model (GPT-style, decoder-only)

---

## Learning Outcomes

After completing this lab you will be able to:

1. Understand sequential data and its characteristics
2. Explain how generative models perform sequence prediction
3. Implement RNN/LSTM and Transformer-based generative models in PyTorch
4. Train models to generate new sequences from learned patterns
5. Evaluate and compare generated sequence quality

---

## File Structure

```
GenAI_Lab_Week9.ipynb   ← Main Colab notebook (all code)
README.md               ← This file
```

---

## Getting Started

### Option 1 — Open directly in Google Colab (recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Select `GenAI_Lab_Week9.ipynb`
4. Click **Runtime → Run all**

No extra installs needed — all dependencies are pre-installed on Colab.

### Option 2 — Run locally

```bash
pip install torch numpy matplotlib
jupyter notebook GenAI_Lab_Week9.ipynb
```

---

## Dataset

A hand-crafted 16-sentence corpus about sequence modeling and deep learning concepts, e.g.:

```
machine learning models learn patterns from data
lstm uses gates to control information flow
sequence generation is widely used in artificial intelligence
...
```

The same corpus is used for both components.

---

## Component I: LSTM Language Model

| Detail | Value |
|---|---|
| Tokenisation | Word-level |
| Input format | Sliding window of 4 tokens → predict next token |
| Architecture | Embedding → 2-layer LSTM → Linear |
| Embedding dim | 64 |
| Hidden dim | 128 |
| Epochs | 200 |
| Optimiser | Adam + StepLR |

**Generation** uses a sliding context window with a tunable **temperature** parameter:

| Temperature | Behaviour |
|---|---|
| 0.5 | Deterministic, conservative |
| 1.0 | Balanced |
| 1.5 | Creative, more surprising |

---

## Component II: Transformer Language Model

| Detail | Value |
|---|---|
| Tokenisation | Word-level with `<SOS>` / `<EOS>` tokens |
| Architecture | Decoder-only Transformer (GPT-style) |
| Positional encoding | Sinusoidal (Vaswani et al., 2017) |
| `d_model` | 64 |
| Attention heads | 4 |
| Layers | 2 |
| Feed-forward dim | 256 |
| Epochs | 300 |
| Optimiser | Adam + CosineAnnealingLR |

**Generation** is auto-regressive — tokens are appended one at a time until `<EOS>` or the max token limit is reached.

---

## Expected Outputs

Running all cells produces:

- Training loss curves for both models
- A side-by-side loss comparison chart
- Generated sequences at three temperature levels (0.5 / 1.0 / 1.5) for multiple seed phrases
- An interactive cell for free-form experimentation

**Sample output (seed: `"lstm uses gates"`):**
```
LSTM        : lstm uses gates to control information flow in many applications
Transformer : lstm uses gates to control information flow and learn patterns
```

---

## Model Comparison

| Aspect | LSTM | Transformer |
|---|---|---|
| Parallelism during training | ❌ Sequential | ✅ Fully parallel |
| Long-range dependencies | Limited | Strong (global attention) |
| Positional awareness | Implicit (recurrence) | Explicit (positional encoding) |
| Generation method | Sliding context window | Auto-regressive append |
| Typical convergence | Faster on small data | May need more epochs |

---

## Key Concepts

**Temperature Sampling** — divides logits before softmax to control randomness. Lower values sharpen the distribution; higher values flatten it.

**Causal Masking** — the Transformer's upper-triangular attention mask prevents each position from attending to future tokens, mimicking the sequential nature of RNNs.

**Positional Encoding** — since Transformers have no inherent sense of order, sinusoidal signals are added to embeddings to encode token position.

---

## References

- Vaswani et al. (2017) — *Attention Is All You Need*
- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*
- PyTorch Documentation — [pytorch.org](https://pytorch.org/docs)
