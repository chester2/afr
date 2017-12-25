# k-nearest neighbors identification.


import numpy as np


def dtoc(fweights, cweights, i, dim):
    # distances to class; get distances between some face (fweights) and all faces in class i
    # tag distances with the class index
    # consider only the first "dim" weights
    if dim == 0:
        return [
            (i, np.linalg.norm(fweights - cweights[:,j]))
            for j in range(cweights.shape[1])]
    else:
        return [
            (i, np.linalg.norm(fweights[:dim] - cweights[:dim,j]))
            for j in range(cweights.shape[1])]


def dtoa(fweights, imset, dim):
    # distances to all training images
    tdists = []
    for i in range(imset.nofc):
        tdists += dtoc(fweights, imset.readcweights(i), i, dim)
    tdists.sort(key=lambda x: x[1])
    return tdists


def identify(tdists, k):
    if k == 1:
        return tdists[0][0]
    counter = {}
    for i, d in tdists:
        try:
            counter[i] += 1
        except:
            counter[i] = 1
        if counter[i] >= k:
            return i