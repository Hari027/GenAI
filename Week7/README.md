# 🎨 Neural Style Transfer — CSET419 Lab 7

Combine the **content** of a photo with the **style** of a painting using a frozen pretrained VGG19. No training — just pixel-level optimization.

> Van Gogh's *Starry Night* painted onto a CIFAR-10 horse image.

---

## How It Works

A generated image starts as a copy of the content image. The optimizer directly updates its pixel values each step to minimize three losses:

| Loss | Layer | Purpose |
|------|-------|---------|
| **Content loss** | `relu4_2` (layer 21) | Preserve shapes and structure |
| **Style loss** | `relu1_1` → `relu5_1` (5 layers) | Match texture via Gram matrices |
| **Total variation loss** | — | Smooth out pixel noise |

**Total loss:** `ALPHA × content + BETA × style + TV_W × tv`

VGG19 weights are **frozen** — only the output image pixels are updated.

---

## Quickstart (Google Colab)

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook
2. Set runtime to GPU: `Runtime > Change runtime type > T4 GPU`
3. Paste the full script into a single cell and run it

No file uploads needed. Everything downloads automatically.

---

## What Gets Downloaded

| Source | What | Size |
|--------|------|------|
| `torchvision` | CIFAR-10 training set | ~170 MB (cached after first run) |
| Wikimedia Commons | Van Gogh – *The Starry Night* (1280px) | ~300 KB |

The script scans CIFAR-10 for the first **horse** image to use as the content image.

---

## Key Parameters

```python
IMG_SIZE = 512      # Output resolution (reduce to 256 on CPU)
ALPHA    = 1.0      # Content loss weight
BETA     = 3e5      # Style loss weight — increase for more painterly effect
TV_W     = 5e-5     # Total variation weight — increase for smoother output
STEPS    = 800      # Outer optimization steps (reduce to 200 on CPU)
```

Optimizer: **L-BFGS** (`lr=1.0`, `max_iter=40`) — more efficient than Adam for NST.

---

## VGG19 Layer Map

```
Layer index   VGG19 name    Role
───────────   ──────────    ────────────────────────────
    0         relu1_1       Style  (weight 1.0)
    5         relu2_1       Style  (weight 0.8)
   10         relu3_1       Style  (weight 0.5)
   19         relu4_1       Style  (weight 0.3)
   21         relu4_2       Content ← main structural layer
   28         relu5_1       Style  (weight 0.2)
```

---

## Output

Saves a 3-panel comparison as `nst_result.png`:

```
[ Content Image ]  [ Style Image ]  [ Stylized Output ]
  CIFAR-10 horse    Starry Night      horse + Van Gogh
```

---

## Runtime

| Hardware | Expected Time |
|----------|--------------|
| GPU (T4) | ~15–25 mins  |
| CPU      | ~45–60 mins (use STEPS=200, IMG_SIZE=256) |

---

## Dependencies

All pre-installed in Google Colab — no `pip install` needed.

```
torch
torchvision
Pillow
matplotlib
requests
```

---

## Course

**CSET419 — Generative AI** | Lab 7: Neural Style Transfer
