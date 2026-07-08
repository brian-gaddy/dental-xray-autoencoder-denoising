"""Dataset loading and reproducible Gaussian-noise utilities."""
from pathlib import Path
import numpy as np

REQUIRED_KEYS = {"x_train", "y_train", "x_test", "y_test"}

def load_dataset(path):
    path = Path(path)
    with np.load(path) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            raise KeyError(f"Missing dataset arrays: {sorted(missing)}")
        return tuple(data[k] for k in ("x_train", "y_train", "x_test", "y_test"))

def add_gaussian_noise(images, noise_factor=0.2, seed=42):
    rng = np.random.default_rng(seed)
    noisy = images + noise_factor * rng.normal(size=images.shape)
    return np.clip(noisy, 0.0, 1.0).astype("float32")
