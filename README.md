# Generative Adversarial Network for High-Resolution Image Synthesis

This project involves designing and training a deep convolutional Generative Adversarial Network (GAN) to synthesize high-resolution facial images. The implementation focuses on architectural robustness and training stability, integrating advanced normalization and progressive training techniques.

## 🔧 Project Overview

The goal of this project was to develop a GAN that could progressively learn to generate realistic human faces by synthesizing high-quality images from low-dimensional noise vectors.

- **Frameworks Used**: PyTorch, CUDA
- **Programming Language**: Python
- **Visualization**: Matplotlib, torchvision

## 🏗️ Architecture

- **Generator**: Deep convolutional architecture with spectral normalization applied to each layer to ensure Lipschitz continuity and stable training dynamics.
- **Discriminator**: Mirrored convolutional structure with LeakyReLU activations and instance normalization to distinguish real from fake images effectively.

## 🔁 Training Pipeline

The training followed a progressive growing strategy:
- Initial image resolution: 64×64
- Gradually scaled up through progressive layers to higher resolutions
- Adaptive training loop with separate optimization steps for the generator and discriminator
- BatchNorm and SpectralNorm applied to stabilize convergence

## 📁 Dataset

The model was trained on a dataset of celebrity face images. The dataset was preprocessed and resized to 64×64 resolution, normalized, and batched for efficient GPU training.

## 🧪 Key Components

- **Spectral Normalization**: Prevents the exploding gradients problem and stabilizes GAN training.
- **Progressive Growing**: Allows smoother learning at lower resolutions before increasing complexity.
- **Loss Function**: Binary Cross-Entropy loss for adversarial training.
- **Optimizers**: Adam with tuned hyperparameters for both networks.

## 🖼️ Training Snapshots

Visual outputs were captured across various training epochs to qualitatively monitor the improvement in image realism and diversity.

## 🔍 Observations

The quality of generated samples improved noticeably across training epochs, reflecting the effectiveness of progressive growing and normalization strategies.

---

This project demonstrates the practical implementation of a progressively trained GAN with spectral normalization to stabilize training and improve image quality.
