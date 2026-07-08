# Dental X-ray Autoencoder Denoising

Deep-learning image restoration project that uses a convolutional denoising autoencoder
to reconstruct cleaner panoramic dental X-rays from synthetically corrupted inputs.

## Project Overview

The workflow loads 256×256 RGB panoramic dental images, injects reproducible Gaussian
noise, trains an encoder-decoder convolutional neural network, and compares noisy inputs
with reconstructed outputs. The project demonstrates image preprocessing, convolutional
representation learning, reconstruction loss optimization, and qualitative model evaluation.

## Technical Stack

Python · TensorFlow/Keras · NumPy · Matplotlib · Pytest · GitHub Actions

## Architecture

**Encoder:** Conv2D(64, stride 2) → Conv2D(32, stride 2)

**Decoder:** Conv2DTranspose(32, stride 2) → Conv2DTranspose(64, stride 2) →
Conv2D(3, sigmoid)

The three-channel output preserves RGB tensor compatibility with the input images.

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── data/raw/
├── notebooks/dental_xray_autoencoder_denoising.ipynb
├── src/data.py
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

Add the dataset as described in `DATA_ACCESS.md`, then run the notebook from the
`notebooks/` directory.

## Modeling Workflow

1. Load the compressed NumPy image arrays.
2. Add Gaussian noise with a factor of 0.2 and clip pixel values to [0, 1].
3. Train the convolutional autoencoder using noisy images as inputs and clean images as targets.
4. Track training and validation reconstruction loss.
5. Encode and decode held-out images.
6. Visually compare noisy and reconstructed dental X-rays.

## Portfolio Value

This repository demonstrates applied computer vision, TensorFlow model subclassing,
convolutional autoencoders, reproducible preprocessing, modular Python design, automated
tests, and CI-ready project structure.

## Responsible Use

This project is for technical demonstration and research-oriented learning. It is not a
medical device and must not be used for diagnosis or clinical decision-making.
