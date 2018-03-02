# Functions for reading and writing images.


import numpy as np
from PIL import Image


def standardize(pixels):
    # convert seq with integer coordinates from 0-255 to a collection of std Gaussian distributed values
    mean = np.mean(pixels)
    std = np.std(pixels)
    return [(p - mean)/std for p in pixels]


def unstandardize(pixels):
    m = min(pixels)
    diff = max(pixels) - m
    return [(p - m)*255/diff for p in pixels]


def imread(fp):
    # open fp, convert to list of pixels, make it greyscale, standardize to std Gaussian distro
    pixels = list(Image.open(fp).convert('L').getdata())
    return standardize(pixels)


def imwrite(fn, pixels, w, h):
    # export list of pixels as a greyscale image to disk
    im = Image.new('L', (w, h),)
    im.putdata(unstandardize(pixels))
    im.save(fn)