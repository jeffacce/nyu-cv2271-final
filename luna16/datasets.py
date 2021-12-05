import numpy as np
import pandas as pd
import os
import scipy.ndimage
import SimpleITK as sitk
from pathlib import Path
import json
from preproc import load_scan, resample
from extract_cubes import coord_to_idx, slice_with_padding, get_slices
from torch.utils.data import Dataset, DataLoader
import itertools


class LUNA16DatasetFromIso(Dataset):
    def __init__(self, iso_root_path, candidates_file, subsets=None):
        self.root = Path(iso_root_path)
        self.candidates = pd.read_csv(self.root / candidates_file)
        with open(self.root / 'uid_to_subset.json') as f:
            self.uid_to_subset = json.load(f)
        self.candidates['subset'] = self.candidates['seriesuid'].apply(self.uid_to_subset.get)
        if subsets is not None:
            self.candidates = self.candidates[self.candidates['subset'].isin(subsets)]
            self.candidates = self.candidates.reset_index(drop=True)
        self.metadata = pd.read_csv(self.root / 'seriesuid_isometric_spacing_origin_direction.csv').set_index('seriesuid')
        self.pos_sample_idx = self.candidates[self.candidates['class'] == 1].index.to_numpy()
        self.neg_sample_idx = self.candidates[self.candidates['class'] == 0].index.to_numpy()
        self.cached_arr = None
        self.cached_seriesuid = None
    
    def __len__(self):
        return len(self.candidates)
    
    def __getitem__(self, idx):
        row = self.candidates.iloc[idx]
        seriesuid = row['seriesuid']
        if (self.cached_seriesuid is not None) and (self.cached_seriesuid == seriesuid):
            arr = self.cached_arr
        else:
            self.cached_seriesuid = seriesuid
            arr = np.load(self.root / self.uid_to_subset[seriesuid] / ('%s.npy' % seriesuid))
            self.cached_arr = arr
        coord = row[['coordX', 'coordY', 'coordZ']].astype(float).to_numpy()
        spacing = self.metadata.loc[seriesuid][:3].to_numpy()
        origin = self.metadata.loc[seriesuid][3:6].to_numpy()
        direction = self.metadata.loc[seriesuid][6:9].to_numpy()
        idx = coord_to_idx(coord, spacing, origin, direction)
        slices = get_slices(idx)
        X = slice_with_padding(slices, arr)
        y = int(row['class'])
        return X, y


class LUNA16DatasetFromCubes(Dataset):
    def __init__(self, cube_root_path, candidates_file, subsets=None):
        self.root = Path(cube_root_path)
        self.candidates = pd.read_csv(self.root / candidates_file)
        self.pos_arr = np.load(self.root / 'pos.npy')
        with open(self.root / 'uid_to_subset.json') as f:
            self.uid_to_subset = json.load(f)
        self.candidates['subset'] = self.candidates['seriesuid'].apply(self.uid_to_subset.get)
        if subsets is not None:
            self.candidates = self.candidates[self.candidates['subset'].isin(subsets)]
            self.candidates = self.candidates.reset_index(drop=True)
        self.pos_sample_idx = self.candidates[self.candidates['class'] == 1].index.to_numpy()
        self.neg_sample_idx = self.candidates[self.candidates['class'] == 0].index.to_numpy()
    
    def __len__(self):
        return len(self.candidates)
    
    def __getitem__(self, idx):
        row = self.candidates.iloc[idx]
        y = int(row['class'])
        if row['class'] == 1:
            X = self.pos_arr[row['i'],:,:,:,:]
        else:
            arr = np.load(self.root / self.uid_to_subset[row['seriesuid']] / ('neg_%s.npy' % row['seriesuid']))
            X = arr[row['i'],:,:,:,:]
        return X, y
