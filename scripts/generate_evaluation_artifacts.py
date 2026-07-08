"""Train/evaluate the dental X-ray denoising autoencoder and save portfolio artifacts.

This script expects the dataset at `data/raw/Dental-Panaromic-Autoencoder.npz`.
Generated outputs:
- data/processed/evaluation_metrics.csv
- data/processed/training_history.csv
- figures/training_loss.png
- figures/training_mae.png
- figures/denoising_examples.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.data import add_gaussian_noise, load_dataset
from src.metrics import mae, mse, psnr, ssim_simple
from src.model import DenoiseAutoencoder

DATA_PATH = Path("data/raw/Dental-Panaromic-Autoencoder.npz")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")


def save_training_plots(history) -> None:
    history_df = pd.DataFrame(history.history)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(PROCESSED_DIR / "training_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="train_loss")
    plt.plot(history_df["val_loss"], label="val_loss")
    plt.title("Autoencoder Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_loss.png", dpi=180)
    plt.close()

    mae_cols = [c for c in history_df.columns if "mae" in c.lower()]
    if mae_cols:
        plt.figure(figsize=(8, 5))
        for col in mae_cols:
            plt.plot(history_df[col], label=col)
        plt.title("Autoencoder Mean Absolute Error")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "training_mae.png", dpi=180)
        plt.close()


def save_example_grid(noisy, reconstructed, clean, n=10) -> None:
    n = min(n, len(clean))
    plt.figure(figsize=(18, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        ax.imshow(noisy[i])
        ax.set_title("Noisy")
        ax.axis("off")

        ax = plt.subplot(3, n, i + 1 + n)
        ax.imshow(reconstructed[i])
        ax.set_title("Denoised")
        ax.axis("off")

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        ax.imshow(clean[i])
        ax.set_title("Clean")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "denoising_examples.png", dpi=180)
    plt.close()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. See DATA_ACCESS.md.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    x_train, _, x_test, _ = load_dataset(DATA_PATH)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    x_train_noisy = add_gaussian_noise(x_train, noise_factor=0.2, seed=42)
    x_test_noisy = add_gaussian_noise(x_test, noise_factor=0.2, seed=43)

    model = DenoiseAutoencoder(input_shape=x_train.shape[1:])
    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"])

    history = model.fit(
        x_train_noisy,
        x_train,
        epochs=50,
        validation_data=(x_test_noisy, x_test),
        verbose=1,
    )
    save_training_plots(history)

    reconstructed = model.predict(x_test_noisy, verbose=1)
    save_example_grid(x_test_noisy, reconstructed, x_test)

    metrics = {
        "test_mse": mse(x_test, reconstructed),
        "test_mae": mae(x_test, reconstructed),
        "test_psnr": psnr(x_test, reconstructed),
        "test_ssim_global": ssim_simple(x_test, reconstructed),
        "test_images": int(len(x_test)),
        "noise_factor": 0.2,
    }
    pd.DataFrame([metrics]).to_csv(PROCESSED_DIR / "evaluation_metrics.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
