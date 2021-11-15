import pytest
import random
import numpy as np
from ..preproc import load_scan, resample


TEST_IMG_PATH = '/scratch/zc2357/cv/final/datasets/luna16/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.979083010707182900091062408058.mhd'


def test_load_scan_axis_order_consistency():
    img, arr = load_scan(TEST_IMG_PATH, clip_min=None, clip_max=None)
    x, y, z = img.GetSize()
    for n in range(10):
        i = random.randint(0, x-1)
        j = random.randint(0, y-1)
        k = random.randint(0, z-1)
        assert arr[i, j, k] == img.GetPixel(i, j, k)


def test_load_scan_min_max_range():
    img, arr = load_scan(TEST_IMG_PATH)
    assert arr.max() <= 3000
    assert arr.min() >= -1000


def test_resample_shape_spacing_consistency():
    img, arr = load_scan(TEST_IMG_PATH)
    arr_iso, spacing_iso = resample(arr, img.GetSpacing())
    arr_orig_resampled, spacing_orig_resampled = resample(arr_iso, spacing_iso, new_spacing=img.GetSpacing())

    assert arr_orig_resampled.shape == arr.shape
    assert np.isclose(spacing_orig_resampled, img.GetSpacing()).all()
