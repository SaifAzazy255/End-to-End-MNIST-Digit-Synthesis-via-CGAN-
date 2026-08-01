# End-to-End MNIST Digit Synthesis via CGAN 🧠

An end-to-end Deep Learning project focused on generative AI to accurately synthesize targeted MNIST handwritten digits. This repository contains the complete pipeline to train a Conditional Generative Adversarial Network (CGAN) from pure random noise to visual inference using PyTorch.

## 🎬 Training Progression
Watch the Generator evolve from producing static noise to creating distinct, sharp handwritten digits:

<img width="602" height="32" alt="cgan_training_progress_v2" src="https://github.com/user-attachments/assets/5151105e-bd4c-4564-9de8-743195701ec2" />

## 🛠️ Project Architecture & Mathematical Balance
This project avoids using standard heavy classification backbones. Instead, it relies on a lightweight, custom-engineered architecture mathematically balanced to avoid checkerboard artifacts for 28x28 grayscale matrices.

*   **Generator:** Utilizes `ConvTranspose2d` layers with specific kernel and stride configurations (e.g., kernel=4, stride=2, padding=1) to upsample from a 1x1 noise seed up to 7x7, 14x14, and finally 28x28 pixels.
*   **Discriminator:** Utilizes `Conv2d` layers to downsample the image back to a single scalar value.
*   **Conditional Embedding:** Integrates `nn.Embedding` directly into the latent space to allow deterministic control over the exact digit class (0-9) generated.

## ⚙️ Adversarial Training Loop
The model is trained using a Min-Max zero-sum game objective implemented via `BCELoss`. The stable training loop ensures that the Generator bypasses the Discriminator without falling into mode collapse, achieving Nash Equilibrium.
