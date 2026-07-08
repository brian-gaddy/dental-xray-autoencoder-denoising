# Dental X-ray Autoencoder Denoising

Deep-learning image restoration project that uses a convolutional denoising autoencoder to reconstruct cleaner panoramic dental X-rays from synthetically corrupted inputs.

## Project Overview

The workflow loads 256×256 RGB panoramic dental images, injects reproducible Gaussian noise, trains an encoder-decoder convolutional neural network, and compares noisy inputs with reconstructed outputs. The project demonstrates image preprocessing, convolutional representation learning, reconstruction loss optimization, quantitative denoising evaluation, and qualitative reconstruction review.

## Technical Stack

Python · TensorFlow/Keras · NumPy · Pandas · Matplotlib · Streamlit · Pytest · GitHub Actions

## Architecture

**Encoder:** Conv2D(64, stride 2) → Conv2D(32, stride 2)

**Decoder:** Conv2DTranspose(32, stride 2) → Conv2DTranspose(64, stride 2) → Conv2D(3, sigmoid)

The three-channel output preserves RGB tensor compatibility with the input images.

## Evaluation Metrics

The project now includes quantitative image-quality evaluation utilities for:

- **MSE** — mean squared reconstruction error
- **MAE** — mean absolute reconstruction error
- **PSNR** — peak signal-to-noise ratio
- **Global SSIM approximation** — lightweight structural-similarity estimate for portfolio reproducibility

After local training, the evaluation pipeline saves:

```text
data/processed/evaluation_metrics.csv
data/processed/training_history.csv
figures/training_loss.png
figures/training_mae.png
figures/denoising_examples.png
```

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── app.py
├── data/raw/
├── notebooks/dental_xray_autoencoder_denoising.ipynb
├── scripts/generate_evaluation_artifacts.py
├── src/data.py
├── src/metrics.py
├── src/model.py
├── tests/
├── DATA_ACCESS.md
├── LICENSE
└── requirements.txt
```

## Quick Start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Add the dataset as described in `DATA_ACCESS.md`, then run the notebook from the `notebooks/` directory.

## Generate Evaluation Artifacts

After placing the dataset locally, run:

```bash
python scripts/generate_evaluation_artifacts.py
```

This trains the autoencoder, saves training curves, generates noisy-vs-denoised-vs-clean examples, and exports PSNR/SSIM/MSE/MAE evaluation metrics.

## Streamlit Review App

Launch the local review app with:

```bash
streamlit run app.py
```

The app displays generated evaluation metrics and image examples when available. It can also compare a clean reference image against a denoised image using MSE, MAE, PSNR, and global SSIM.

## Modeling Workflow

1. Load the compressed NumPy image arrays.
2. Add Gaussian noise with a factor of 0.2 and clip pixel values to [0, 1].
3. Train the convolutional autoencoder using noisy images as inputs and clean images as targets.
4. Track training and validation reconstruction loss and MAE.
5. Encode and decode held-out images.
6. Calculate PSNR, SSIM, MSE, and MAE.
7. Visually compare noisy, denoised, and clean dental X-rays.

## Portfolio Value

This repository demonstrates applied computer vision, TensorFlow model subclassing, convolutional autoencoders, reproducible preprocessing, reconstruction-quality metrics, image artifact generation, Streamlit review tooling, modular Python design, automated tests, and CI-ready project structure.

## Responsible Use

This project is for technical demonstration and research-oriented learning. It is not a medical device and must not be used for diagnosis or clinical decision-making.
