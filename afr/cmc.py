# Class mean (average) classification.


import numpy as np


def dtocm(fweights, imset, dim):
    # distances to class means
    cmeans = imset.readcmeans()
    if dim == 0:
        return [
            np.linalg.norm(fweights - cmeans[:,j])
            for j in range(cmeans.shape[1])]
    else:
        return [
            np.linalg.norm(fweights[:dim] - cmeans[:,j][:dim])
            for j in range(cmeans.shape[1])]


def identify(dists):
    return dists.index(min(dists))