# Functions for reading and writing images.

from PIL import Image

def imread(filepath):
    # open fp, convert to vector, make it greyscale, return the vector
    return list(Image.open(filepath).convert('L').getdata())

def imwrite(filename, vector, w, h):
    # export vector as greyscale image and save to disk
    im = Image.new('L', (w, h),)
    im.putdata([round(i) for i in vector])
    im.save(filename)