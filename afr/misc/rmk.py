# Functions for exporting images.


import os
import numpy as np
from ..prereq.consts import *
from ..prereq.imio import *
from ..prereq.path import *
from ..prereq.dbio import *
from ..core.eigf import f_to_weights


#####################################
## Helper Function
#####################################


def get_setinfo(setname):
    mean = readmean(setname)
    eigfs = readeigfs(setname)
    w = SETS[setname][WIDTH]
    h = SETS[setname][HEIGHT]
    return mean, eigfs, w, h


#####################################
## Main Functions
#####################################


@pathreset
def rmk_img(filepath, setname):
    #filepath is absolute
    filename = os.path.split(filepath)[1]
    fweights = f_to_weights(filepath, setname)
    mean, eigfs, w, h = get_setinfo(setname)
    face = eigfs.dot(fweights) + mean
    pathexports()
    imwrite(f'rmk_{filename}', face, w, h)


@pathreset
def rmk_mean(setname):
    # export the mean as an image
    mean, eigfs, w, h = get_setinfo(setname) # eigfs isn't needed
    pathexports()
    imwrite(f'{setname}_mean.png', mean, w, h)


@pathreset
def rmk_eigfs(setname):
    # export all setname's eigenfaces to for human-viewing
    mean, eigfs, w, h = get_setinfo(setname)
    pathexports()
    for j in range(eigfs.shape[1]):
        # here, 100 was arbitrarily chosen such that it gave a nice-looking result
        f = 100 / max(eigfs[:,j], key=lambda x:abs(x))
        imwrite(f'{setname}_eigf{j}.png', f*eigfs[:,j] + mean, w, h)


@pathreset
def rmk_cmeans(setname):
    # exports the average of each class
    mean, eigfs, w, h = get_setinfo(setname)
    cmeans = readcmeans(setname)
    pathexports()
    for j in range(cmeans.shape[1]):
        face = eigfs.dot(cmeans[:,j]) + mean
        imwrite(f'{setname}_cmean{j}.png', face, w, h)