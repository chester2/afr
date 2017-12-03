# k-nearest neighbors identification.


import os
import numpy as np
from ..prereq.consts import *
from ..prereq.imio import *
from ..prereq.path import *
from ..prereq.dbio import *
from .eigf import f_to_weights
from ..misc.rmk import rmk_img


#####################################
## Helper Functions
#####################################


def dists_to_class(fweights, setname, i, dim):
    # get distances between some face (fweights) and all faces in class i
    # mark distances with the class index
    # consider only the first "dim" weights
    cweights = readclass(setname, i)
    if dim == 0:
        return [
            (i, np.linalg.norm(fweights - cweights[:,j]))
            for j in range(cweights.shape[1])
        ]
    else:
        return [
            (i, np.linalg.norm(fweights[:dim] - cweights[:dim,j]))
            for j in range(cweights.shape[1])
        ]


def dists_to_all(fweights, setname, dim):
    pathdb()
    files = [
        filename
        for filename in ls()
        if f'{setname}_class' in filename
    ]
    files.sort()
    dists = []
    for i, filename in enumerate(files):
        dists += dists_to_class(fweights, setname, i, dim)
    dists.sort(key=lambda x: x[1])
    return dists


def id(fweights, setname, k, dim):
    dists = dists_to_all(fweights, setname, dim)
    if k > 1:
        n_classes = SETS[setname][END] - SETS[setname][INIT] + 1
        tracker = [0] * n_classes
        for d in dists:
            tracker[d[0]] += 1
            if tracker[d[0]] >= k:
                return d[0]
    else:
        return dists[0][0]


#####################################
## Main Function
#####################################


@pathreset
def knn(filename, setname, k, dim=0, rmk=False):
    filepath = os.path.join(PATHTBI, filename)
    fweights = f_to_weights(filepath, setname)
    if rmk:
        rmk_img(filepath, setname)
    idindex = id(fweights, setname, k, dim)
    if SETS[setname][NAMES]:
        name = SETS[setname][NAMES][idindex]
        return idindex, name
    return idindex, ''