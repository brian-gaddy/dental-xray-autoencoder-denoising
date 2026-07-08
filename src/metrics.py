"""Image reconstruction metrics for denoising autoencoders."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared reconstruction error."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.square(y_true - y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean absolute reconstruction error."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))


def psnr(y_true: np.ndarray, y_pred: np.ndarray, max_pixel: float = 1.0) -> float:
    """Return peak signal-to-noise ratio in decibels."""
    error = mse(y_true, y_pred)
    if error == 0:
        return float("inf")
    return float(20 * np.log10(max_pixel / np.sqrt(error)))


def ssim_simple(y_true: np.ndarray, y_pred: np.ndarray, max_pixel: float = 1.0) -> float:
    """Return a lightweight global SSIM approximation.

    This implementation is dependency-light for CI and portfolio reproducibility. For
    publication-grade image-quality analysis, compare with a windowed SSIM implementation
    such as scikit-image's structural_similarity.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    c1 = (0.01 * max_pixel) ** 2
    c2 = (0.03 * max_pixel) ** 2

    mu_x = y_true.mean()
    mu_y = y_pred.mean()
    sigma_x = y_true.var()
    sigma_y = y_pred.var()
    sigma_xy = ((y_true - mu_x) * (y_pred - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator)
