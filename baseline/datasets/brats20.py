from .utils import pad_to_multiples_of_n, linear_transform_to_0_1, SegmentationDataset, get_random_largest_fit_slice
import torch
import numpy as np


SLICE_SIZES = np.arange(4, 6) * 16

def transform_t1(arr):
    arr = pad_to_multiples_of_n(arr, n=(16, 16, 1), pad_val=0)
    arr = linear_transform_to_0_1(arr, 0, 2200)
    return arr


def transform_t2(arr):
    arr = pad_to_multiples_of_n(arr, n=(16, 16, 1), pad_val=0)
    arr = linear_transform_to_0_1(arr, 0, 3000)
    return arr


def transform_t1ce(arr):
    arr = pad_to_multiples_of_n(arr, n=(16, 16, 1), pad_val=0)
    arr = linear_transform_to_0_1(arr, 0, 3000)
    return arr


def transform_flair(arr):
    arr = pad_to_multiples_of_n(arr, n=(16, 16, 1), pad_val=0)
    arr = linear_transform_to_0_1(arr, 0, 1500)
    return arr


def transform_y(arr):
    arr = pad_to_multiples_of_n(arr, n=(16, 16, 1), pad_val=0)
    arr = torch.Tensor(arr)
    arr = torch.clamp(arr, min=0, max=1)
    return arr


def transform_X_y(X, y):
    try:
        axial_start, axial_end = get_random_largest_fit_slice(X.shape[3], SLICE_SIZES)
        result_X = X[:,:,:,axial_start:axial_end]
        result_y = y[:,:,:,axial_start:axial_end]
    except:
        result_X = pad_to_multiples_of_n(X, n=(1, 1, 1, SLICE_SIZES[0]), pad_val=0)
        result_y = pad_to_multiples_of_n(y, n=(1, 1, 1, SLICE_SIZES[0]), pad_val=0)
    return result_X, result_y


def transform_X_y_no_slice(X, y):
    result_X = pad_to_multiples_of_n(X, n=(1, 1, 1, 16), pad_val=0)
    result_y = pad_to_multiples_of_n(y, n=(1, 1, 1, 16), pad_val=0)
    return result_X, result_y


brats20 = SegmentationDataset(
    '/scratch/zc2357/cv/final/datasets/brats20/nii/train',
    ['*t1.nii','*t2.nii','*t1ce.nii','*flair.nii'],
    ['*seg.nii'],
    transforms={
        '*t1.nii': transform_t1,
        '*t2.nii': transform_t2,
        '*t1ce.nii': transform_t1ce,
        '*flair.nii': transform_flair,
        '*seg.nii': transform_y,
    },
    transform_X_y=transform_X_y,
)

brats20_no_slice = SegmentationDataset(
    '/scratch/zc2357/cv/final/datasets/brats20/nii/train',
    ['*t1.nii','*t2.nii','*t1ce.nii','*flair.nii'],
    ['*seg.nii'],
    transforms={
        '*t1.nii': transform_t1,
        '*t2.nii': transform_t2,
        '*t1ce.nii': transform_t1ce,
        '*flair.nii': transform_flair,
        '*seg.nii': transform_y,
    },
    transform_X_y=transform_X_y_no_slice,
)
