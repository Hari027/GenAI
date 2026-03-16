# 🎨 Artistic Outputs using GANs — CSET419 Lab 8

Generate artistic images by exploring the latent space of two GAN architectures — a **DCGAN trained from scratch** on CIFAR-10, and a **pretrained BigGAN** for high-quality conditional generation.

---

## What This Lab Covers

| Task | Implementation |
|------|---------------|
| Basic GAN | DCGAN trained on CIFAR-10 |
| Advanced GAN | Pretrained BigGAN-128 (conditional) |
| Latent space exploration | 10 random artistic samples |
| Latent interpolation | SLERP (DCGAN) + linear (BigGAN) |

---

## How It Works

### Part 1 — DCGAN (Basic GAN)

A Deep Convolutional GAN is trained from scratch on CIFAR-10. Two networks compete:

- **Generator** — takes a random noise vector (latent dim = 100) and upsamples it through transposed convolutions into a 64×64 RGB image
- **Discriminator** — tries to tell real CIFAR-10 images from fakes

Neither network ever sees a "correct answer" — they only learn by playing against each other.

```
Latent vector (100,)  →  Generator  →  64×64 image
                                            ↓
                                     Discriminator  →  real / fake
```

Training runs for **5 epochs** (~4–5 mins on GPU).

### Part 2 — Latent Space Exploration

After training, the generator's weights are frozen and used to:

- **Generate 10 random artistic samples** by sampling different noise vectors
- **Interpolate between two latent vectors** using SLERP (spherical linear interpolation) — produces smoother transitions than basic linear interpolation

### Part 3 — BigGAN (Advanced GAN)

Uses a pretrained `biggan-deep-128` model from HuggingFace. No training needed — just load and generate. BigGAN is **conditional**, meaning you pass a class label alongside the noise vector to control what gets generated.

Classes used: `cliff`, `volcano`, `coral reef`, `starfish`, `lakeside`, `coral fungus`

---

## Output Files

| File | Contents |
|------|----------|
| `dcgan_samples.png` | 10 artistic images from random DCGAN latent vectors |
| `dcgan_interpolation.png` | 10-step SLERP walk between two latent points |
| `biggan_samples.png` | 6 BigGAN images, one per class |
| `biggan_interpolation.png` | 8-step linear interpolation for "volcano" class |

---

## Quickstart (Google Colab)

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook
2. Set runtime to GPU: `Runtime > Change runtime type > T4 GPU`
3. Paste the full script into a single cell and run

No file uploads needed. Everything downloads automatically.

---

## Key Parameters

```python
# DCGAN
LATENT_DIM  = 100     # size of the input noise vector
IMAGE_SIZE  = 64      # output image resolution
EPOCHS      = 5       # increase for better quality (20+ for sharp results)
BATCH_SIZE  = 128
LR          = 0.0002  # Adam learning rate
BETA1       = 0.5     # Adam momentum term

# BigGAN
TRUNCATION  = 0.4     # lower = higher quality, less diversity (range: 0.02–1.0)
```

---

## DCGAN Architecture

**Generator** (latent → image)
```
100×1×1  →  512×4×4  →  256×8×8  →  128×16×16  →  64×32×32  →  3×64×64
            ConvT+BN     ConvT+BN     ConvT+BN       ConvT+BN      Tanh
```

**Discriminator** (image → real/fake)
```
3×64×64  →  64×32×32  →  128×16×16  →  256×8×8  →  512×4×4  →  1
             Conv+LReLU   Conv+BN+LReLU  Conv+BN     Conv+BN     Sigmoid
```

| Model | Parameters |
|-------|-----------|
| Generator | 3,576,704 |
| Discriminator | 2,765,568 |

---

## Dependencies

All pre-installed in Google Colab except `pytorch-pretrained-biggan` (auto-installed by the script).

```
torch
torchvision
matplotlib
numpy
nltk
pytorch-pretrained-biggan   ← installed automatically by the script
```

> The script also auto-downloads the NLTK `wordnet` and `omw-1.4` corpora required by BigGAN's class name lookup.

---

## Runtime

| Hardware | Expected Time |
|----------|--------------|
| GPU (T4) | ~5–8 mins total |
| CPU | Not recommended |

---

## Notes

- DCGAN output at 5 epochs will look rough — that is expected. GANs need many more epochs for sharp images. The goal is to observe the learning process and latent space behavior.
- BigGAN output is high quality since it uses a pretrained model with no training required.
- SLERP interpolation is used for DCGAN because latent vectors lie on a hypersphere — spherical interpolation respects that geometry better than linear blending.

---

## Course

**CSET419 — Introduction to Generative AI** | Lab 8: Artistic Outputs using Neural Art Concepts
