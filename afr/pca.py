# Determine the principle components (eigenfaces) and their eigenvalues, as well as the mean training image.


import os
import numpy as np
from .imio import *
from .pathreset import pathreset


@pathreset
def import_ims(dir_to_ims):
    # import all training images from the setname folder and return a row matrix of imported face images
    os.chdir(dir_to_ims)
    m = [imread(filename) for filename in os.listdir()]
    return np.array(m, dtype=np.float64)


def build_reqs(rmatrix):
    mean = np.mean(rmatrix, 0)
    AT = rmatrix - mean
    A = AT.transpose()
    L = AT.dot(A)
    eigvs, eigfs = np.linalg.eigh(L)
    eigfs = A.dot(eigfs)
    # sort eigvs in descending order
    eigvs = np.flipud(eigvs)
    # ensure eigvs[i] corresponds to eigfs[:,i]
    eigfs = np.fliplr(eigfs)
    return mean, eigvs, eigfs


def normalize(eigvs, eigfs):
    # normalize eigfs and modify eigvs accordingly
    for j in range(eigfs.shape[1]):
        factor = np.linalg.norm(eigfs[:,j])
        if factor > 0:
            eigvs[j] *= factor
            for i in range(eigfs.shape[0]):
                eigfs[i,j] /= factor


def pca(imset):
    rmatrix = import_ims(imset.dir_to_ims)
    mean, eigvs, eigfs = build_reqs(rmatrix)
    normalize(eigvs, eigfs)
    imset.writemean(mean)
    imset.writeeigvs(eigvs)
    imset.writeeigfs(eigfs)