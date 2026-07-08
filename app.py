"""Streamlit demo for dental X-ray denoising review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.metrics import mae, mse, psnr, ssim_simple

st.set_page_config(page_title="Dental X-ray Autoencoder Denoising", layout="wide")
st.title("Dental X-ray Autoencoder Denoising")
st.caption("Portfolio demo for reviewing noisy, denoised, and clean panoramic dental X-ray images.")

metrics_path = Path("data/processed/evaluation_metrics.csv")
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test MSE", f"{metrics['test_mse']:.5f}")
    c2.metric("Test MAE", f"{metrics['test_mae']:.5f}")
    c3.metric("PSNR", f"{metrics['test_psnr']:.2f} dB")
    c4.metric("Global SSIM", f"{metrics['test_ssim_global']:.3f}")
else:
    st.info("Run `python scripts/generate_evaluation_artifacts.py` after placing the dataset locally to generate evaluation metrics.")

st.subheader("Denoising Example Artifact")
example_path = Path("figures/denoising_examples.png")
if example_path.exists():
    st.image(str(example_path), caption="Noisy vs denoised vs clean dental X-ray examples", use_container_width=True)
else:
    st.info("No example grid found yet. Generate it with the evaluation script after local model training.")

st.subheader("Single Image Metric Comparison")
st.write("Upload a clean reference image and a denoised/reconstructed image to calculate image-quality metrics.")
clean_file = st.file_uploader("Clean/reference image", type=["png", "jpg", "jpeg"], key="clean")
denoised_file = st.file_uploader("Denoised/reconstructed image", type=["png", "jpg", "jpeg"], key="denoised")

if clean_file and denoised_file:
    clean = np.asarray(Image.open(clean_file).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    denoised = np.asarray(Image.open(denoised_file).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0

    left, right = st.columns(2)
    left.image(clean, caption="Clean/reference", use_container_width=True)
    right.image(denoised, caption="Denoised/reconstructed", use_container_width=True)

    st.write(
        {
            "mse": mse(clean, denoised),
            "mae": mae(clean, denoised),
            "psnr": psnr(clean, denoised),
            "ssim_global": ssim_simple(clean, denoised),
        }
    )

st.subheader("Responsible Use")
st.write("This app is for technical portfolio review only. It is not a diagnostic or clinical decision-support tool.")
