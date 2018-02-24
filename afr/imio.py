# Functions for reading and writing images.

from PIL import Image

def imread(fp):
    # open fp, convert to vector, make it greyscale, return the vector
    return list(Image.open(fp).convert('L').getdata())

def imwrite(fn, vector, w, h):
    # export vector as greyscale image and save to disk
    im = Image.new('L', (w, h),)
    im.putdata([round(i) for i in vector])
    im.save(fn)