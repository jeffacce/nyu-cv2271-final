# https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial

import numpy as np
import pandas as pd
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_scan(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    # air is -1000. < -1000 is out of scanner bounds (essentially NaN); clip to -1000
    # > 3000 is foreign body: metal implants, pacemakers, etc.
    arr = arr.clip(min=-1000, max=3000)
    arr = np.moveaxis(arr, 0, -1)
    return img, arr


def resample(arr, orig_spacing, new_spacing=[1,1,1]):
    orig_spacing = np.array(orig_spacing)
    new_spacing = np.array(new_spacing)

    resize_factor = orig_spacing / new_spacing
    new_shape = np.round(arr.shape * resize_factor)
    real_resize_factor = new_shape / arr.shape
    new_spacing = orig_spacing / real_resize_factor

    result = scipy.ndimage.interpolation.zoom(arr, real_resize_factor, mode='nearest')
    result = result.clip(arr.min(), arr.max())
    return result, new_spacing


def plot_3d(image, threshold=300):
    verts, faces, normals, vals = measure.marching_cubes(image, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])

    plt.show()
