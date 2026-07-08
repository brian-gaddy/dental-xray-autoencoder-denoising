import numpy as np

from src.metrics import mae, mse, psnr, ssim_simple


def test_metrics_return_expected_values_for_identical_images():
    x = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    assert mse(x, x) == 0.0
    assert mae(x, x) == 0.0
    assert psnr(x, x) == float("inf")
    assert ssim_simple(x, x) > 0.99


def test_metrics_detect_difference():
    x = np.zeros((4, 4, 3), dtype=np.float32)
    y = np.ones((4, 4, 3), dtype=np.float32)
    assert mse(x, y) == 1.0
    assert mae(x, y) == 1.0
    assert psnr(x, y) == 0.0
