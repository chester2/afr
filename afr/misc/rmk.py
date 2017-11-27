# Functions for exporting images.


import os
import numpy as np
from ..prereq.consts import *
from ..prereq.imio import *
from ..prereq.path import *
from ..prereq.dbio import *
from ..core.eigf import f_to_weights


#####################################
## Helper Functions
#####################################


def lincombo(scalars, colmatrix):
    # returns a linear combination of colmatrix's columns
    # scalars[j] corresponds to colmatrix[:,j]
    vector = scalars[0]*colmatrix[:,0]
    for j in range(1, colmatrix.shape[1]):
        vector += scalars[j]*colmatrix[:,j]
    return vector


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
def rmk_img(setname, filepath):
    #filepath is absolute
    filename = os.path.split(filepath)[1]
    fweights = f_to_weights(setname, filepath)
    mean, eigfs, w, h = get_setinfo(setname)
    vector = lincombo(fweights, eigfs) + mean
    pathexports()
    imwrite(f'rmk_{filename}', vector, w, h)


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
        vector = list(lincombo(cmeans[:,j], eigfs) + mean)
        imwrite(f'{setname}_cmean{j}.png', vector, w, h)