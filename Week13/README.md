# 🧠 Generative AI Model with Reinforcement Learning

A hands-on Jupyter notebook lab that builds a self-improving text generation system from scratch using a Markov-chain policy and the REINFORCE algorithm — no deep learning frameworks required.

---

## 📋 Overview

The agent learns to generate domain-relevant sentences (climate-tech headlines) purely through trial and reward — no labelled training data. It follows the classic RL loop:

```
Generate sentence → Compute reward → Update policy → Repeat
```

This mirrors the conceptual foundation behind real-world systems like RLHF (the technique used to fine-tune ChatGPT) and recommendation engines.

---

## 🗂️ File

| File | Description |
|------|-------------|
| `GenAI_RL_Lab_modified.ipynb` | Main Colab-ready notebook |

---

## 🚀 Getting Started

### Run in Google Colab (recommended)

1. Download `GenAI_RL_Lab_modified.ipynb`
2. Go to [colab.research.google.com](https://colab.research.google.com)
3. Click **File → Upload notebook** and select the file
4. Click **Runtime → Run all**

### Run locally

```bash
pip install numpy matplotlib seaborn
jupyter notebook GenAI_RL_Lab_modified.ipynb
```

---

## 🧩 Lab Structure

| Step | Section | What It Does |
|------|---------|--------------|
| — | Imports | Loads numpy, matplotlib, seaborn; sets random seeds |
| 1 | Dataset | Defines 5 climate-tech reference sentences |
| 2 | Vocabulary | Builds word ↔ index mappings from the corpus |
| 3 | Generative Model | Initialises the Markov-chain policy matrix + temperature sampling |
| 4 | Reward Function | Jaccard similarity against the closest reference sentence |
| 5 | Before RL | Generates 5 sentences with the untrained policy |
| 6 | RL Training | 300-episode REINFORCE loop with live logging |
| 7 | After RL | Generates 5 sentences with the trained policy |
| 8 | Results | Summary stats + 3-panel visualisation |

---

## 🔑 Key Concepts

**Generative AI** — sentences are produced by sampling a Markov-chain transition matrix `policy[i][j]`, where each entry is the probability of word `j` following word `i`. This is the discrete analogue of how GPT samples the next token.

**Reinforcement Learning (REINFORCE)** — after each generated sentence the policy is nudged toward transitions that earned a higher reward, with no labelled examples needed.

**Jaccard Similarity** — the reward metric. Defined as `|intersection| / |union|` of word sets. Stricter than simple overlap because it penalises both missing words and extra words.

**Temperature Sampling** — a scalar applied before sampling that controls diversity. Values below 1 sharpen the distribution (more deterministic); values above 1 flatten it (more random).

---

## 📊 Visualisations

The results cell produces a 3-panel figure:

- **Reward curve** — episode-by-episode Jaccard reward with a 20-episode moving average
- **Before vs After bar chart** — average reward comparison
- **Policy heatmap** — learned transition probabilities for the top-8 start words, showing which word-to-word paths the agent reinforced

---

## ⚙️ Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `lr` | `0.15` | Learning rate for policy gradient updates |
| `episodes` | `300` | Total training episodes |
| `temperature` | `0.8` | Generation sharpness during training |
| `window` | `20` | Moving-average window for the reward plot |

---

## 🔬 Extensions to Try

- **Change temperature** — try `0.3` (focused) vs `1.5` (diverse) and re-run training
- **Expand the dataset** — add more sentences and observe how the vocabulary and policy matrix grow
- **Swap the reward** — replace Jaccard with BLEU score for a standard NLP metric
- **Longer sentences** — increase the `length` parameter in `generate_sentence()` and see how coherence changes

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Runtime |
| NumPy | any | Policy matrix operations |
| Matplotlib | any | Reward curve + bar chart |
| Seaborn | any | Policy heatmap |

All dependencies come pre-installed in Google Colab.

---

## 🎓 Learning Outcomes

By completing this lab you will be able to:

- Explain how a Markov-chain model acts as a simple generative AI
- Implement the REINFORCE policy gradient algorithm from scratch
- Design a reward function and understand its effect on agent behaviour
- Interpret a policy transition matrix as a heatmap
- Connect this toy framework to production systems like RLHF
