# Class mean (average) classification.


import os
import numpy as np
from ..prereq.consts import *
from ..prereq.imio import *
from ..prereq.path import *
from ..prereq.dbio import * 
from .eigf import f_to_weights
from ..misc.rmk import rmk_img


#####################################
## Helper Function
#####################################


def dists_to_cmeans(fweights, setname, dim):
    cmeanweights = readcmeans(setname)
    if dim == 0:
        return [
            np.linalg.norm(fweights - cmeanweights[:,j])
            for j in range(cmeanweights.shape[1])
        ]
    else:
        return [
            np.linalg.norm(fweights[:dim] - cmeanweights[:,j][:dim])
            for j in range(cmeanweights.shape[1])
        ]


#####################################
## Main Function
#####################################


@pathreset
def cmc(filename, setname, dim=0, rmk=False):
    filepath = os.path.join(PATHTBI, filename)
    fweights = f_to_weights(filepath, setname)
    if rmk:
        rmk_img(filepath, setname)
    dists = dists_to_cmeans(fweights, setname, dim)
    idindex = dists.index(min(dists))
    if SETS[setname][NAMES]:
        return idindex, SETS[setname][NAMES][idindex]
    return idindex, ''