
# 🧠 Generative Adversarial Network for High-Resolution Image Synthesis

This project implements a deep convolutional GAN (DCGAN) with **spectral normalization** and **progressive growing** to generate high-quality facial images. The focus of this implementation is to develop a stable and scalable GAN architecture capable of generating increasingly realistic samples over training epochs.

## 🚀 Project Overview

- **Goal:** Synthesize photo-realistic face images by training a progressively growing GAN from lower to higher resolutions.
- **Frameworks Used:** Python, PyTorch, CUDA, NumPy, Matplotlib
- **Current Output Resolution:** 64×64 (progressive layers integrated up to this resolution)
- **Next Milestone:** Extend progressive growing to reach 256×256 resolution and beyond.

## 🏗️ Architecture and Training Details

- **Base Model:** Deep Convolutional GAN (DCGAN)
- **Stabilization Techniques:**
  - Spectral Normalization on both Generator and Discriminator
  - Progressive growing of layers during training
- **Loss Function:** Non-saturating GAN loss with regularization
- **Training Platform:** CUDA-enabled GPU

## 📈 Training Procedure

1. **Start with 4×4 resolution** and gradually double it through 8×8, 16×16, 32×32 up to 64×64.
2. Apply **fade-in** transitions when adding new layers to both Generator and Discriminator.
3. Normalize feature maps using **spectral normalization** to avoid exploding gradients and stabilize training.
4. Use **Adam optimizer** with tuned learning rates for Generator and Discriminator.

## 🧪 Monitoring & Visualization

Epoch-wise progress was tracked and visualized to monitor image fidelity and diversity.
Future metrics such as **Fréchet Inception Distance (FID)** will be included to quantify quality improvement.

## 📌 Next Steps

- Scale the generator to output **128×128** and **256×256** resolution images.
- Introduce perceptual loss for enhanced feature realism.
- Integrate evaluation metrics such as **FID** and **IS** (Inception Score).

## 📂 Repository Structure

```
├── data/                  # Training data
├── models/                # Generator and Discriminator definitions
├── training/              # Training scripts and config
├── outputs/               # Epoch-wise generated image samples
├── utils/                 # Helper functions (e.g., progressive interpolation, normalization)
└── README.md              # Project overview
```

## 🙌 Acknowledgments

Built as part of a simulated project for portfolio development. Inspired by the techniques in ProGAN and Spectral Normalization GANs.

---

Feel free to reach out for collaboration or suggestions on extending the project to real-time inference or additional datasets!
