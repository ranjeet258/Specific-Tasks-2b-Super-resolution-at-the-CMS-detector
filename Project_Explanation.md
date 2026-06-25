# CMS Calorimeter Super-Resolution GAN: Project Explanation

*This document is a comprehensive breakdown of my project, **Specific Task 2b: Super Resolution at the CMS Detector**, summarizing the core challenges, architectural decisions, and final results.*

---

## 1. Project Overview & Problem Statement
**The Goal:** The aim of my project is to super-resolve (upscale) CMS calorimeter jet images from a low resolution of **64×64** to a high resolution of **125×125**.

**Why it matters:** Higher resolution images provide finer details of particle energy deposits. This is crucial for improving downstream physics tasks, specifically for better **quark and gluon jet classification** in high-energy physics.

## 2. The Dataset & The Core Challenge
**The Data:** 
- I used the CMS QCD Jets dataset. The images consist of 3 channels representing energy deposits in the **ECAL** (Electromagnetic Calorimeter), **HCAL** (Hadron Calorimeter), and **Tracker**.
- **The Core Challenge (Sparsity):** Calorimeter images are extremely sparse—about **98% of the pixels are zero**. Standard Super-Resolution (SR) models, like a vanilla SRGAN, fail completely here. Because the image is mostly zeros, standard loss functions (like MSE or L1) incentivize the model to simply output all zeros, leading to a "collapse" where the model predicts an entirely blank image.

## 3. Architectural Design & Solutions
To solve the sparsity problem and enforce physical constraints, I implemented custom architectural designs. While I built three different generator architectures (Modified SRGAN, Swin-Transformer, and a Conditional Diffusion model), my core baseline is the **Physics-informed SRGAN**.

### Key Design Decisions (How I solved the challenges):
*   **The LR Skip Connection (Crucial Fix):** 
    *   *What I did:* Instead of having the generator predict the entire HR image from scratch, I upsampled the original Low-Resolution (LR) image using bicubic interpolation and added it to the generator's output (`F.interpolate(LR) + output`).
    *   *Why I did it:* This forces the model to learn the *residual details* (the enhancement) rather than reconstructing the sparse background. This single change completely solved the all-zero collapse problem.
*   **Residual Dense Blocks (RRDB) with SE Attention:** 
    *   *What I did:* I used 12 dense blocks combined with Squeeze-and-Excitation (SE) channel attention.
    *   *Why I did it:* Dense connections ensure maximum feature reuse, which is essential for preserving tiny, sparse signals. SE attention allows the model to treat the ECAL, HCAL, and Tracker channels independently, scaling them based on their distinct physical properties.
*   **Solving the "Non-Integer" Upsampling Problem (64 to 125):**
    *   *What I did:* Since 125 is not an integer multiple of 64, I couldn't simply use a standard 2x upsample. Instead, I first upsampled 64×64 to 128×128 using `PixelShuffle(×2)`, and then applied an `AdaptiveAvgPool2d(125)` to downscale it precisely to 125×125.
    *   *Why I did it:* `PixelShuffle` avoids the checkerboard artifacts caused by transposed convolutions. `AdaptiveAvgPool` ensures an *exact* 125×125 output without floating-point interpolation rounding errors, smoothly removing the 3 boundary pixels.
*   **Spectral-Norm PatchGAN Discriminator:** I used this to ensure stable adversarial training on the highly sparse backgrounds.

## 4. Physics-Informed Loss Function
Standard GAN loss wasn't enough; the model needed to obey the laws of physics. 
My custom loss function combined:
1.  **MSE Loss (1.0):** For basic pixel fidelity.
2.  **LSGAN Adversarial Loss (0.005):** For texture sharpness and realistic jet shapes.
3.  **Energy Conservation Loss (2.0):** A custom physics constraint ensuring the total energy in the super-resolved image matches the high-resolution ground truth.
4.  **Jet Profile Loss (0.05):** Ensures the spatial shape and distribution of the jet are preserved.
*Note: I explicitly disabled L1 Sparsity Loss (set to 0.0) because it was pushing the already sparse outputs to zero.*

## 5. Training Strategy & Engineering
*   **Two-Phase Training:** I pretrained the generator for 3 epochs using only MSE loss to establish a stable baseline, then trained the full GAN (Generator + Discriminator) for 6 epochs using the full physics-informed loss.
*   **Optimization:** I used the Adam optimizer (lr=1e-4) with a CosineAnnealingLR scheduler for smooth learning rate decay.
*   **Hardware & Memory Efficiency:** I trained the model on a Tesla T4 GPU using PyTorch Mixed Precision (FP16 / AMP), which saved ~2x the VRAM and allowed for efficient batch processing with a batch size of 8.

## 6. Final Results & Impact
My model successfully super-resolves the jets while strictly preserving the physics properties. 
*   **Metrics:** I achieved an overall **PSNR of 35.79 dB** and an **SSIM of 0.9406**, showing highly accurate reconstructions.
*   **Physics Metrics:** I achieved a low Peak Shift (2.09 pixels), meaning the core of the energy deposits did not drift spatially during the upsampling process.
