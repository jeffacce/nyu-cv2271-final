import numpy as np
import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset, SequentialSampler
import itertools
import nibabel as nib
from math import ceil
from natsort import natsorted
import torch


def get_slices(idx, size):
    if len(idx) != len(size):
        raise ValueError('Dimension mismatch: idx and size must have the same length.')
    result = []
    for i in range(len(idx)):
        start = idx[i] - size[i]//2
        end = start + size[i]
        result.append((start, end))
    return result


def slice_with_padding(slices, arr, pad_val=-1000):
    '''
    Slices an array with constant padding if the slice is out of bounds.

    Example:
    - axis 0 has `[0, 360)`
    - slice is `[-1, 47)`
    - actual slice should be `[0, 47)`
    - padding should be `(1, 0)` pixels to the `(left, right)`

    Example:
    - axis 1 has `[0, 360)`
    - slice is `[320, 368)`
    - actual slice should be `[320, 360)`
    - padding should be `(0, 8)` pixels to the `(left, right)`
    '''
    assert len(slices) == len(arr.shape)
    
    slices_actual = []
    padding = []
    for i in range(len(slices)):
        this_slice_actual = (max(0, slices[i][0]), min(arr.shape[i], slices[i][1]))
        this_padding = (this_slice_actual[0] - slices[i][0], slices[i][1] - this_slice_actual[1])
        slices_actual.append(this_slice_actual)
        padding.append(this_padding)
    
    cube = arr[slice(*slices_actual[0]), slice(*slices_actual[1]), slice(*slices_actual[2])]
    cube = np.pad(cube, pad_width=padding, mode='constant', constant_values=pad_val)
    return cube


def pad_to_multiples_of_n(arr, n=(16, 16, 1), pad_val=-1000):
    if type(n) is tuple:
        if len(n) != len(arr.shape):
            raise ValueError('Dimension mismatch: n must be the same length as arr.shape')
        else:
            shape_desired = [ceil(arr.shape[i] / n[i]) * n[i] for i in range(len(arr.shape))]
    elif type(n) is int:
        shape_desired = [ceil(x/n) * n for x in arr.shape]
    else:
        raise ValueError('n must be int (for all dimensions) or tuple (specify n for each dimension)')
    
    median_idx = [x // 2 for x in arr.shape]
    slices = get_slices(median_idx, size=shape_desired)
    result = slice_with_padding(slices, arr, pad_val=pad_val)
    if type(arr) is torch.Tensor:
        return torch.Tensor(result)
    else:
        return result


def get_random_largest_fit_slice(input_size, allowed_sizes):
    allowed_sizes = np.array(sorted(allowed_sizes))
    if input_size < allowed_sizes.min():
        raise ValueError('Input size too small (%s) for the smallest allowed size (%s)' % (input_size, allowed_sizes[0]))
    else:
        chosen_size = allowed_sizes[np.where(input_size >= allowed_sizes)[0].max()]
        start = np.random.randint(0, input_size - chosen_size)
        end = start + chosen_size
    return (start, end)


def linear_transform_to_0_1(X, min, max):
    result = torch.clamp(torch.Tensor(X), min=min, max=max)
    result = result - min
    result = result / (max - min)
    return result


class SegmentationDataset(Dataset):
    def __init__(self, root_path: os.PathLike, input_filename_patterns: list, target_filename_patterns: list, transforms: dict, transform_X_y=None):
        '''
            root_path: dataset root directory path.
                The directory should contain a number of subdirectories, each representing one sample.
                Each sample directory should contain input_filenames and target_filenames, all in `.nii` format.
            input_filename_patterns: a list of input filename patterns to match and read from, ordered by channel.
                Example: ['case_*_T1.nii', 'case_*_T2.nii', 'case_*_FLAIR.nii', 'case_*_T1CE.nii']
            target_filename_patterns: a list of output target filename patterns to match and read from, ordered by channel.
                Example: ['case_*_organ_ground_truth.nii', 'case_*_tumor_ground_truth.nii']
            transforms: transform functions for each channel.
            transform_X_y: transform function for both (X, y): function(X, y) -> (X, y)
                Performs a transform on (X, y). E.g. data augmentation, slicing.
        '''
        for key in transforms:
            if key not in (input_filename_patterns + target_filename_patterns):
                raise ValueError('%s in transforms not found in any of the patterns.' % key)
        self.root = Path(root_path)
        self.input_filename_patterns = input_filename_patterns
        self.target_filename_patterns = target_filename_patterns
        self.sample_list = natsorted([x for x in self.root.glob('*') if x.is_dir()])
        self.transforms = transforms
        self.transform_X_y = transform_X_y
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        def _match_pattern(path, pattern):
            fpath = list(path.glob(elem))
            if len(fpath) > 1:
                raise ValueError('Filename match not unique: %s' % fpath)
            elif len(fpath) == 0:
                raise ValueError('No match for pattern: %s' % elem)
            else:
                return fpath[0]
        
        sample_path = self.sample_list[idx]
        X = []
        y = []
        for elem in self.input_filename_patterns:
            fpath = _match_pattern(sample_path, elem)
            img = nib.load(fpath.as_posix())
            arr = img.get_fdata()
            f_transform = self.transforms.get(elem)
            if f_transform is not None:
                arr = f_transform(arr)
            X.append(torch.Tensor(arr))
        for elem in self.target_filename_patterns:
            fpath = _match_pattern(sample_path, elem)
            img = nib.load(fpath.as_posix())
            arr = img.get_fdata()
            f_transform = self.transforms.get(elem)
            if f_transform is not None:
                arr = f_transform(arr)
            y.append(torch.Tensor(arr))
        X = torch.stack(X)
        y = torch.stack(y)
        if self.transform_X_y is not None:
            X, y = self.transform_X_y(X, y)
        return X, y
