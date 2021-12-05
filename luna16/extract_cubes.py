import numpy as np
import pandas as pd
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib import patches
import SimpleITK as sitk
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import json
from preproc import load_scan, resample
from tqdm import tqdm


def coord_to_idx(coord, spacing, origin, direction):
    return (((coord - origin) / spacing) * direction).round().astype(int)


def get_slices(idx, size=48):
    result = []
    for i in range(len(idx)):
        start = idx[i] - size//2
        end = start + size
        result.append((start, end))
    return result


# !BUG? Does matplotlib imshow plot x/y axes differently than expected?
# The slicing seems to work the correct order (xyz), but imshow seems to show it in yxz.
# hard coded 48 slices
def visualize_annotation(arr, idx):
    plt.figure(figsize=(24, 18))
    for i in range(48):
        z_offset = i - 24
        ax = plt.subplot(6, 8, i+1)
        # this yxz order is pretty weird. Double check
        rect = patches.Rectangle((idx[1]-24, idx[0]-24), 48, 48, linewidth=1, edgecolor='r', facecolor='none')
        plt.axhline(idx[0])
        plt.axvline(idx[1])
        ax.add_patch(rect)
        plt.imshow(arr[:,:,idx[2] + z_offset], cmap='gray')
    plt.show()


def assert_in_bounds(slices, arr, verbose=False):
    assert len(slices) == len(arr.shape), 'Dimensions do not match: %s dim for slices, %s dim for arr' % (len(slices), len(arr.shape))
    for i in range(len(arr.shape)):
        start, end = slices[i]
        if not (0 <= start < arr.shape[i]):
            if verbose:
                print('start %s out of bounds [0, %s)' % (start, arr.shape[i]))
            return False
        if not (0 <= end < arr.shape[i]):
            if verbose:
                print('end %s out of bounds [0, %s)' % (end, arr.shape[i]))
            return False
    return True


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
def slice_with_padding(slices, arr, pad_val=-1000):
    assert len(slices) == len(arr.shape) == 3
    
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


if __name__ == '__main__':
    ROOT_ISO = Path('/scratch/zc2357/cv/final/datasets/luna16_iso')
    ROOT_CUBES = Path('/scratch/zc2357/cv/final/datasets/luna16_cubes')
    if not ROOT_CUBES.exists():
        ROOT_CUBES.mkdir()

    with open(ROOT_ISO / 'uid_to_subset.json') as f:
        uid_to_subset = json.load(f)


    candidates = pd.read_csv(ROOT_ISO / 'candidates_V2.csv').set_index('seriesuid')
    metadata = pd.read_csv(ROOT_ISO / 'seriesuid_isometric_spacing_origin_direction.csv').set_index('seriesuid')
    pos_cubes = []
    pos_cubes_metadata = []


    for seriesuid, subset_idx in tqdm(uid_to_subset.items()):
        path = ROOT_ISO / subset_idx / ('%s.npy' % seriesuid)
        print(path)

        candidates_case = candidates.loc[seriesuid]
        spacing = metadata.loc[seriesuid].to_numpy()[:3]
        origin = metadata.loc[seriesuid].to_numpy()[3:6]
        direction = metadata.loc[seriesuid].to_numpy()[6:]
        arr = np.load(path.as_posix())

        neg_cubes = []
        for i in range(len(candidates_case)):
            row = candidates_case.iloc[i]
            coord = row[['coordX', 'coordY', 'coordZ']].astype(float).to_numpy()
            label = row['class'].astype(int)
            idx = coord_to_idx(coord, spacing, origin, direction)
            slices = get_slices(idx)
            cube = slice_with_padding(slices, arr, pad_val=-1000)
            if label == 1:
                pos_cubes.append(cube)
                pos_cubes_metadata.append([seriesuid, *idx])
            else:
                neg_cubes.append(cube)

        if not (ROOT_CUBES / subset_idx).exists():
            (ROOT_CUBES / subset_idx).mkdir()

        neg_cubes = np.stack(neg_cubes).reshape(-1, 1, 48, 48, 48)
        neg_savepath = ROOT_CUBES / subset_idx / ('neg_%s.npy' % seriesuid)
        np.save(neg_savepath, neg_cubes)

    pos_cubes = np.stack(pos_cubes).reshape(-1, 1, 48, 48, 48)
    pos_savepath = ROOT_CUBES / 'pos.npy'
    np.save(pos_savepath, pos_cubes)

    pos_cubes_metadata = pd.DataFrame(pos_cubes_metadata)
    pos_cubes_metadata.columns = ['seriesuid', 'idxX', 'idxY', 'idxZ']
    pos_cubes_metadata.to_csv(ROOT_CUBES / 'pos_cubes_metadata.csv', index=False)
