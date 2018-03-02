# Operations related to finding eigenfaces (principal components) and generating face weights.


import os

import numpy as np


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


def pca(rmatrix):
    mean, eigvs, eigfs = build_reqs(rmatrix)
    normalize(eigvs, eigfs)
    return mean, eigvs, eigfs


def ptow(pixels, mean, eigfs):
    # pixels to weights
    mean_shifted = np.array(pixels, dtype=np.float64) - mean
    return eigfs.transpose().dot(mean_shifted)