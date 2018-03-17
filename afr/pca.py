# Operations related to finding eigenfaces (principal components) and generating face weights.


import numpy as np
from numba import njit


def build_reqs(rmatrix):
    mean = np.mean(rmatrix, 0)
    AT = rmatrix - mean
    A = AT.T
    L = AT @ (A)
    eigvs, eigfs = np.linalg.eigh(L)
    eigfs = A @ eigfs
    # sort eigvs in descending order
    eigvs = np.flipud(eigvs)
    # ensure eigvs[i] corresponds to eigfs[:,i]
    eigfs = np.fliplr(eigfs)
    # make eigvs and eigfs C_CONTIGUOUS again
    return mean, np.ascontiguousarray(eigvs), np.ascontiguousarray(eigfs)


@njit
def normalize(eigvs, eigfs):
    # normalize eigfs and modify eigvs accordingly
    for j in range(eigfs.shape[1]):
        factor = np.linalg.norm(eigfs[:,j])
        if factor > 0:
            eigvs[j] *= factor
            for i in range(eigfs.shape[0]):
                eigfs[i,j] /= factor


def pca(rmatrix):
    mean, eigvs, eigfs = build_reqs(rmatrix)
    normalize(eigvs, eigfs)
    return mean, eigvs, eigfs


def ptow(pixels, mean, eigfs):
    # pixels to weights
    return eigfs.T @ (pixels - mean)