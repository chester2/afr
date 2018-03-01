# Functions for reading and writing images.

import numpy as np
from PIL import Image

def standardize(pixels):
    mean = np.mean(pixels)
    std = np.std(pixels)
    for i in range(len(pixels)):
        pixels[i] = (pixels[i] - mean) / std

def imread(fp):
    # open fp, convert to list of pixels, make it greyscale
    pixels = list(Image.open(fp).convert('L').getdata())
    standardize(pixels)
    return pixels

def imwrite(fn, pixels, w, h):
    # export list of pixels as a greyscale image to disk
    im = Image.new('L', (w, h),)
    im.putdata(pixels)
    im.save(fn)