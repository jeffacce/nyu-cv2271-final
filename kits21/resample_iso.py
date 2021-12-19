import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import scipy
import scipy.ndimage
from pathlib import Path
from tqdm import tqdm


def load_scan(path, clip_min=-1000, clip_max=3000):
    path = Path(path).as_posix()  # cast to str
    img = nib.load(path)
    arr = img.get_data()
    if not ((clip_min is None) and (clip_max is None)):
        # air is -1000. < -1000 is out of scanner bounds (essentially NaN); clip to -1000
        # > 3000 is foreign body: metal implants, pacemakers, etc.
        arr = arr.clip(min=clip_min, max=clip_max)
    arr = arr.transpose(2, 1, 0)
    return img, arr

# only valid for KiTS 2021
def get_spacing(img):
    return -img.affine[0,2], -img.affine[1,1], -img.affine[2,0]

def resample(arr, orig_spacing, new_spacing=(1,1,1), order=3, cval=-1024):
    orig_spacing = np.array(orig_spacing)
    new_spacing = np.array(new_spacing)

    resize_factor = orig_spacing / new_spacing
    new_shape = np.round(arr.shape * resize_factor)
    real_resize_factor = new_shape / arr.shape
    new_spacing = orig_spacing / real_resize_factor

    result = scipy.ndimage.interpolation.zoom(arr, real_resize_factor, order=order, mode='constant', cval=cval)
    result = result.clip(arr.min(), arr.max())
    return result, new_spacing

def show_slice_overlay(img_slice, seg_slice, alpha=0.5):
    plt.figure(dpi=300)
    plt.imshow(img_slice, cmap='gray')
    plt.imshow(seg_slice, cmap='inferno', alpha=alpha)
    plt.show()


ROOT = Path('/scratch/zc2357/cv/final/datasets/kits21/kits21/data')
ROOT_SAVE = Path('/scratch/zc2357/cv/final/datasets/kits21_iso')


if __name__ == '__main__':
    for i in tqdm(range(300)):
        case = ROOT / ('case_%05d' % i)
        case_save = ROOT_SAVE / ('case_%05d' % i)
        if not case_save.exists():
            case_save.mkdir()

        print(case.stem)

        img_path = case / 'imaging.nii.gz'
        seg_path = case / 'aggregated_MAJ_seg.nii.gz'

        print('\t load')
        img, img_arr = load_scan(img_path)
        seg, seg_arr = load_scan(seg_path)

        print('\t resample')
        spacing = get_spacing(img)
        img_arr_iso, img_new_spacing = resample(img_arr, spacing, new_spacing=[1,1,1], order=3, cval=-1000)
        seg_arr_iso, seg_new_spacing = resample(seg_arr, spacing, new_spacing=[1,1,1], order=0, cval=0)

        assert np.isclose(img_new_spacing, seg_new_spacing).all()
        assert img_arr_iso.shape == seg_arr_iso.shape

        # from LiTS preproc: only select slices with segmentation,
        # ±20 slices top and bottom
        EXPAND_SLICE = 20
        start_slice, end_slice = np.where(seg_arr_iso.any(axis=(0,1)))[0][[0, -1]]
        end_slice += 1
        start_slice = max(0, start_slice - EXPAND_SLICE)
        end_slice = min(seg_arr_iso.shape[0], end_slice + EXPAND_SLICE)
        img_arr_iso = img_arr_iso[:,:, start_slice:end_slice]
        seg_arr_iso = seg_arr_iso[:,:, start_slice:end_slice]
        print('\t total slices:', end_slice - start_slice)
        print('\t shape:', img_arr_iso.shape)

        print('\t save')
        img_iso = nib.Nifti1Image(img_arr_iso, np.diag((*img_new_spacing, 1)))
        seg_iso = nib.Nifti1Image(seg_arr_iso, np.diag((*seg_new_spacing, 1)))
        img_savepath = (case_save / 'imaging.nii.gz').as_posix()
        seg_savepath = (case_save / 'aggregated_MAJ_seg.nii.gz').as_posix()
        img_iso.to_filename(img_savepath)
        seg_iso.to_filename(seg_savepath)
