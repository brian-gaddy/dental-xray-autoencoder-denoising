import numpy as np
from src.data import add_gaussian_noise

def test_noise_is_clipped_and_reproducible():
    x = np.full((2, 4, 4, 3), 0.5, dtype=np.float32)
    a = add_gaussian_noise(x, seed=7)
    b = add_gaussian_noise(x, seed=7)
    assert np.array_equal(a, b)
    assert a.min() >= 0.0
    assert a.max() <= 1.0
