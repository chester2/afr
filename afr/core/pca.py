# Functions for constructing a eigenfaces and their eigenvalues.


import numpy as np
from ..prereq.consts import *
from ..prereq.imio import *
from ..prereq.path import *
from ..prereq.dbio import *


#####################################
## Helper Functions
#####################################


def import_training_imgs(setname):
    # import all traning images from the setname folder and return a row matrix of imported face images
    pathset(setname)
    m = [imread(filename) for filename in ls()]
    return np.array(m, dtype=np.float64)


def build_reqs(faces_rmatrix):
    mean = np.mean(faces_rmatrix, 0)
    AT = faces_rmatrix - mean
    A = np.transpose(AT)
    L = AT.dot(A)
    eigvs, eigfs = np.linalg.eigh(L)
    eigfs = A.dot(eigfs)
    eigvs = np.flipud(eigvs)
    eigfs = np.fliplr(eigfs)
    return mean, A, eigvs, eigfs


def normalize(A, eigvs, eigfs):
    # normalizes eigfs and modifies eigvs accordingly
    for j in range(eigfs.shape[1]):
        factor = np.linalg.norm(eigfs[:,j])
        eigvs[j] *= factor
        for i in range(eigfs.shape[0]):
            eigfs[i,j] /= factor


#####################################
## Main Function
#####################################


@pathreset
def pca(setname):
    faces_rmatrix = import_training_imgs(setname)
    mean, A, eigvs, eigfs = build_reqs(faces_rmatrix)
    normalize(A, eigvs, eigfs)
    writemean(setname, mean)
    writeeigvs(setname, eigvs)
    writeeigfs(setname, eigfs)