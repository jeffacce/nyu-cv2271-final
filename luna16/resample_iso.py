import pandas as pd
import numpy as np
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
import json
from preproc import load_scan, resample
from tqdm import tqdm


ROOT_RAW = Path('/scratch/zc2357/cv/final/datasets/luna16')
ROOT_ISO = Path('/scratch/zc2357/cv/final/datasets/luna16_iso')

with open(ROOT_RAW / 'uid_to_subset.json') as f:
    uid_to_subset = json.load(f)

if not ROOT_ISO.exists():
    ROOT_ISO.mkdir()

df = []
for seriesuid, subset_idx in tqdm(uid_to_subset.items()):
    loadpath = ROOT_RAW / subset_idx / (seriesuid + '.mhd')
    img, arr = load_scan(loadpath.as_posix())
    originX, originY, originZ = img.GetOrigin()
    arr_iso, spacing_iso = resample(arr, img.GetSpacing())
    spacingX, spacingY, spacingZ = spacing_iso
    row = [seriesuid, spacingX, spacingY, spacingZ, originX, originY, originZ]
    df.append(row)
    
    if not (ROOT_ISO / subset_idx).exists():
        (ROOT_ISO / subset_idx).mkdir()
    
    savepath = ROOT_ISO / subset_idx / (seriesuid + '.npy')
    np.save(savepath, arr_iso)

df = pd.DataFrame(df)
df.columns = ['seriesuid', 'spacingX', 'spacingY', 'spacingZ', 'originX', 'originY', 'originZ']
df.to_csv(ROOT_ISO / 'seriesuid_isometric_spacing_origin.csv', index=False)
