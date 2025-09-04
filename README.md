# SRCNN-Reimplementation-on-DIV2K-Datasets

This repository contains a reimplementation of the **SRCNN (Super-Resolution Convolutional Neural Network)** for image super-resolution, trained and evaluated on **DIV2K Datasets**.  
It is a PyTorch-based implementation that demonstrates high-quality upscaling and performance evaluation with PSNR and SSIM metrics.

## Models

You can download trained SRCNN models (trained on DIV2K (x3 only)) here:

- [SRCNN x3 (Google Drive)](https://drive.google.com/drive/folders/1Nv-ZU3OpHA9f7KTrQeWfWpL1c6jH8w-Q?usp=sharing)

## Features

- Reimplementation of the original SRCNN architecture
- Supports training and testing on **DIV2K Datasets**
- Evaluation with PSNR and SSIM metrics
- Save and load model checkpoints
- Example scripts for training and inference

## 📊 Example Outputs

<p align="center">
  <img src="assets/srcnn_baseline.png" alt="SRCNN Baseline" width="45%"/>
  <img src="assets/srcnn_improved.png" alt="SRCNN Improved" width="45%"/>
</p>

<p align="center">
  <em>Left: Baseline SRCNN &nbsp;&nbsp; | &nbsp;&nbsp; Right: Improved SRCNN</em>
</p>
