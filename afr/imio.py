# Functions for reading and writing images.


import numpy as np
from numba import njit
from PIL import Image


@njit
def standardize(pixels):
    # convert np array with coordinates from 0-255 to a collection of std Gaussian distributed values
    mean = np.mean(pixels)
    std = np.std(pixels)
    return np.float64([(p - mean)/std for p in pixels])


@njit
def unstandardize(pixels):
    # shift and scale pixels to fill the range 0-255
    m = np.min(pixels)
    diff = np.max(pixels) - m
    return [(p - m)*255/diff for p in pixels]


def imread(fp):
    # open fp, convert to list of pixels, make it greyscale, standardize to std Gaussian distro
    pixels = np.float64(Image.open(fp).convert('L').getdata())
    return standardize(pixels)


def imwrite(fp, pixels, w, h):
    # export list of pixels as a greyscale image to disk
    im = Image.new('L', (w, h),)
    im.putdata(unstandardize(pixels))
    im.save(fp)