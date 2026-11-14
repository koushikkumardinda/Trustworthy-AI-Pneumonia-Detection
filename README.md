# Trustworthy AI for Multi-Class Pneumonia Detection

This repository contains the official implementation of my M.Tech research project focused on building a trustworthy and explainable AI framework for medical imaging. The pipeline classifies chest X-rays into three categories (Normal, Bacterial Pneumonia, and Viral Pneumonia) while providing uncertainty quantification and visual counterfactual explanations.

## Abstract
Deep learning models in healthcare often lack transparency and struggle with overconfidence in ambiguous cases. This project addresses these critical gaps by combining evidential deep learning with generative counterfactuals. By generating a "pseudo-healthy" version of an infected X-ray, the model produces a difference map that highlights the exact regions of infection, providing clinicians with interpretable and reliable diagnostic support.

## Methodology
The framework operates in four distinct phases:

* **Phase 1: Environment & Dataset Setup**
  * Utilizes aggressive data augmentation (flips, rotations, color jitter) on a multi-class pneumonia dataset.
  * Optimized for T4 GPU environments using Automatic Mixed Precision (AMP).
* **Phase 2: Evidential Uncertainty Model (Swin-EViT)**
  * Replaces the standard classifier of a pretrained Swin Transformer (Base) with an Evidential Head.
  * Uses Expected Cross Entropy (ECE) Loss and KL Divergence Regularization to quantify prediction uncertainty.
* **Phase 3: Generative XAI (CycleGAN)**
  * Employs a CycleGAN architecture with a GeneratorResNet and a PatchGAN-style Discriminator.
  * Generates "pseudo-healthy" counterfactual images from infected X-rays (Normal to Pneumonia and vice versa).
* **Phase 4: Evaluation & Difference Map Extraction**
  * Evaluates confidence using Expected Calibration Error (ECE) and image generation quality via Fréchet Inception Distance (FID).
  * Subtracts the generated pseudo-healthy image from the original infected X-ray to isolate the disease pathology in a clear "Difference Map".

## Prerequisites
Ensure the following libraries are installed to run the notebook:
```bash
pip install torch torchvision timm torchmetrics[image] torch-fidelity scikit-learn matplotlib numpy pandas pillow
