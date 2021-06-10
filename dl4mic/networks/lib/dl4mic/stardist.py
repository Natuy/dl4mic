from __future__ import print_function
# Keras imports
from keras import models
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger # we currently don't use any other callbacks from ModelCheckpoints
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as keras

# General import

import numpy as np
import os
import glob
from skimage import img_as_ubyte, io, transform
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from pathlib import Path
import shutil
import random
import time
import csv
import sys
from math import ceil
from tifffile import imread, imsave

# Imports for QC
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image
from scipy import signal
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from skimage.util import img_as_uint
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr


from sklearn.feature_extraction import image
from skimage import img_as_ubyte, io, transform
from skimage.util.shape import view_as_windows

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available, export_imagej_rois
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D
from csbdeep.utils import Path, normalize, download_and_extract_zip_file, plot_history # for loss plot
from csbdeep.io import save_tiff_imagej_compatible

# Suppressing some warnings
import warnings
warnings.filterwarnings('ignore')



def weighted_binary_crossentropy(class_weights):

    def _weighted_binary_crossentropy(y_true, y_pred):
        binary_crossentropy = keras.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * class_weights[1] + (1. - y_true) * class_weights[0]
        weighted_binary_crossentropy = weight_vector * binary_crossentropy

        return keras.mean(weighted_binary_crossentropy)

    return _weighted_binary_crossentropy


# Normalization functions from Martin Weigert
def normalizePercentile(x, pmin=1, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):#dtype=np.float32
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x



# Simple normalization to min/max fir the Mask
def normalizeMinMax(x, dtype=np.float32):
  x = x.astype(dtype,copy=False)
  x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
  return x

def stardist(config,name,basedir):
    return StarDist2D(config, name=name, basedir=basedir)


def saveResult(save_path, nparray, polygons, source_dir_list, prefix=''):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for (filename, image,poly) in zip(source_dir_list, nparray, polygons):
        imsave(os.path.join(save_path, prefix+os.path.splitext(filename)[0]+'.tif'), image, polygons)
        export_imagej_rois(os.path.join(save_path, prefix+os.path.splitext(filename)[0]), poly['coord'])


def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

def export_imagej_rois(fname, polygons, set_position=True, subpixel=True, compression=ZIP_DEFLATED):
    """ polygons assumed to be a list of arrays with shape (id,2,c) """

    if isinstance(polygons,np.ndarray):
        polygons = (polygons,)

    if polygons[0].shape[0]>0:
        fname = Path(fname)
        if fname.suffix == '.zip':
            fname = fname.with_suffix('')

        with ZipFile(str(fname)+'.zip', mode='w', compression=compression) as roizip:
            for pos,polygroup in enumerate(polygons,start=1):
                for i,poly in enumerate(polygroup,start=1):
                    roi = polyroi_bytearray(poly[1],poly[0], pos=(pos if set_position else None), subpixel=subpixel)
                    roizip.writestr('{pos:03d}_{i:03d}.roi'.format(pos=pos,i=i), roi)


def polyroi_bytearray(x,y,pos=None,subpixel=True):
    """ Byte array of polygon roi with provided x and y coordinates
        See https://github.com/imagej/imagej1/blob/master/ij/io/RoiDecoder.java
    """
    import struct
    def _int16(x):
        return int(x).to_bytes(2, byteorder='big', signed=True)
    def _uint16(x):
        return int(x).to_bytes(2, byteorder='big', signed=False)
    def _int32(x):
        return int(x).to_bytes(4, byteorder='big', signed=True)
    def _float(x):
        return struct.pack(">f", x)

    subpixel = bool(subpixel)
    # add offset since pixel center is at (0.5,0.5) in ImageJ
    x_raw = np.asarray(x).ravel() + 0.5
    y_raw = np.asarray(y).ravel() + 0.5
    x = np.round(x_raw)
    y = np.round(y_raw)
    assert len(x) == len(y)
    top, left, bottom, right = y.min(), x.min(), y.max(), x.max() # bbox
    n_coords = len(x)
    bytes_header = 64
    bytes_total = bytes_header + n_coords*2*2 + subpixel*n_coords*2*4
    B = [0] * bytes_total
    B[ 0: 4] = map(ord,'Iout')   # magic start
    B[ 4: 6] = _int16(227)       # version
    B[ 6: 8] = _int16(0)         # roi type (0 = polygon)
    B[ 8:10] = _int16(top)       # bbox top
    B[10:12] = _int16(left)      # bbox left
    B[12:14] = _int16(bottom)    # bbox bottom
    B[14:16] = _int16(right)     # bbox right
    B[16:18] = _uint16(n_coords) # number of coordinates
    if subpixel:
        B[50:52] = _int16(128)   # subpixel resolution (option flag)
    if pos is not None:
        B[56:60] = _int32(pos)   # position (C, Z, or T)

    for i,(_x,_y) in enumerate(zip(x,y)):
        xs = bytes_header + 2*i
        ys = xs + 2*n_coords
        B[xs:xs+2] = _int16(_x - left)
        B[ys:ys+2] = _int16(_y - top)

    if subpixel:
        base1 = bytes_header + n_coords*2*2
        base2 = base1 + n_coords*4
        for i,(_x,_y) in enumerate(zip(x_raw,y_raw)):
            xs = base1 + 4*i
            ys = base2 + 4*i
            B[xs:xs+4] = _float(_x)
            B[ys:ys+4] = _float(_y)

    return bytearray(B)

# -------------- Other definitions -----------
prediction_prefix = ''

