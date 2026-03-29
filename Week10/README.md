# CSET419 – Introduction to Generative AI
## Lab 10: Sequential Data Generation

---

## Objective

Implement generative models that learn patterns from sequential data and generate new sequences using **RNN/LSTM** and **Transformer** architectures.

---

## Learning Outcomes

After completing this lab, you will be able to:

1. Understand the concept of sequential data generation
2. Preprocess sequence datasets for deep learning models
3. Implement generative models for sequence prediction
4. Train neural network models to generate new sequences
5. Evaluate the quality and coherence of generated sequences

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

All packages come pre-installed in Google Colab. No additional setup needed.

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `GenAI_Lab_Week10.ipynb`
3. Run all cells top to bottom (`Runtime → Run all`)

---

## Structure

```
GenAI_Lab_Week10.ipynb
│
├── Setup & Imports
├── Dataset (16-sentence corpus, shared across both components)
│
├── Component I – LSTM Based Sequential Data Generation
│   ├── Task 1 & 2 – Preprocessing & Character-Level Tokenization
│   ├── Task 3   – Input-Output Sequence Pairs
│   ├── Task 4   – LSTM Model Design
│   ├── Task 5   – Training
│   └── Task 6   – Text Generation (seed → output)
│
├── Component II – Transformer Based Sequential Data Generation
│   ├── Task 2 – Word-Level Tokenization
│   ├── Task 3 – Positional Encoding
│   ├── Task 4 – Transformer Encoder Architecture
│   ├── Task 5 – Training & Text Generation
│
└── Comparison – LSTM vs Transformer (loss curves + summary table)
```

---

## Models

### Component I – LSTM
| Parameter | Value |
|---|---|
| Tokenization | Character-level (whitespace split → char index) |
| Sequence length | 30 characters |
| Embedding dim | 64 |
| Hidden size | 128 |
| Layers | 2 |
| Epochs | 100 |
| Optimizer | Adam + StepLR |

### Component II – Transformer
| Parameter | Value |
|---|---|
| Tokenization | Word-level (whitespace tokenization) |
| Sequence length | 8 words |
| Model dim (d_model) | 64 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Feedforward dim | 256 |
| Epochs | 150 |
| Optimizer | Adam + CosineAnnealingLR |

---

## Expected Output

**Component I** — Generated character sequences from a seed string, e.g.:

```
Seed: "deep learning"
Generated: "deep learning models improve sequence learning generative mod..."
```

**Component II** — Generated word sequences from a seed word list, e.g.:

```
Seed: "sequence generation is"
Output: "sequence generation is used in chatbots and assistants machine learning..."
```

---

## Notes

- The dataset is intentionally small (16 sentences) so training completes in seconds on CPU.
- Generated text quality is limited by dataset size — this is a toy model for learning purposes.
- Temperature controls output randomness: lower = conservative, higher = creative.
- Both models use a causal (left-to-right) generation strategy.

---

## Key Concepts

| Concept | Description |
|---|---|
| Sequential Data Generation | Learning patterns from ordered data to produce new sequences |
| Character-level Tokenization | Each character is a token |
| Word-level Tokenization | Each word is a token |
| Positional Encoding | Injects position information into Transformer inputs |
| Causal Mask | Prevents the model from attending to future tokens |
| Temperature Sampling | Controls randomness during text generation |
