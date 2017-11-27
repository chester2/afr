# Functions for constructing a eigenfaces and their eigenvalues.


import numpy as np
from ..prereq.consts import *
from ..prereq.imio import *
from ..prereq.path import *
from ..prereq.dbio import *


#####################################
## Helper Functions
#####################################


def get_training(setname):
    # import all traning images from the setname folder and return a row matrix of imported face images
    pathset(setname)
    m = [imread(filename) for filename in ls()]
    return np.array(m, dtype=np.float64)


def build_reqs(db):
    # parameter db is a row matrix of face images
    # mean is the mean face image vector
    # A is a column matrix of mean-shifted face images
    # AT is the transpose of A
    # L is AT*A, or the lower-dimensional covariance matrix
    # w is a list of eigenvalues of L sorted in increasing order
    # v is a column matrix of normalized eigenvectors of L; v[:,j] corresponds to w[j]
    mean = np.mean(db, 0)
    AT = db - mean
    A = np.transpose(AT)
    L = AT.dot(A)
    l, v = np.linalg.eigh(L)
    return mean, A, l, v


def normalize_u(A, l, v):
    # returns a 2-tuple
    # eigvs is a list of normalized eigenvalues of A*AT sorted in descending order
    # eigfs is a column matrix of normalized eigenvectors of A*AT's (u is exactly equal to A*v; u[:,j] corresponds to l[j])
    temp = A.dot(v)
    eigvs = []
    eigfs = []
    for j in range(temp.shape[1]):
        c = np.linalg.norm(temp[:,j])
        eigvs.append(l[j] * c)
        eigfs.append(temp[:,j]/c)
    eigvs.reverse()
    eigfs.reverse()
    return eigvs, np.transpose(np.array(eigfs, dtype=np.float64))


#####################################
## Main Function
#####################################


@pathreset
def pca(setname):
    # main PCA process
    db = get_training(setname)
    mean, A, l, v = build_reqs(db)
    eigvs, eigfs = normalize_u(A, l, v)
    writemean(setname, mean)
    writeeigvs(setname, eigvs)
    writeeigfs(setname, eigfs)