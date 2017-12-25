# Determine eigenface weights.


import os
import re
import numpy as np
from .imio import *
from .pathreset import pathreset


@pathreset
def build_fclasses(imset):
    # returns a 3D list (class, face/image, pixel intensity)
    os.chdir(imset.dir_to_ims)
    fclasses = []
    files = os.listdir()
    for i in range(imset.ifirst, imset.ifinal + 1):
        fclasses.append([
            imread(filename)
            for filename in files
            if re.match(imset.fnid(i), filename)])
    return fclasses


def build_cweights(fclass, mean, eigfs):
    # returns a column matrix of weights of fclass's faces
    mean_shifted = (np.array(fclass, dtype=np.float64) - mean).transpose()
    return eigfs.transpose().dot(mean_shifted)


def build_cmeans(list_of_cweights):
    return np.array(
        [cweights.mean(1) for cweights in list_of_cweights],
        dtype=np.float64).transpose()


def build_fweights(face, mean, eigfs):
    # returns a vector of face's weights
    # face is a list
    mean_shifted = np.array(face, dtype=np.float64) - mean
    return eigfs.transpose().dot(mean_shifted)


def eigf(imset):
    mean = imset.readmean()
    eigfs = imset.readeigfs()
    fclasses = build_fclasses(imset)
    list_of_cweights = []
    for i, fclass in enumerate(fclasses):
        cweights = build_cweights(fclass, mean, eigfs)
        imset.writecweights(i, cweights)
        list_of_cweights.append(cweights)
    cmeans = build_cmeans(list_of_cweights)
    imset.writecmeans(cmeans)