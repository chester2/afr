# k-nearest neighbor classification.


import numpy as np


def dtoti(weights, subset):
    # list of (timg, dist) elements
    return [
        (timg, np.linalg.norm(weights - timg.weights))
        for timg in subset]


def knn(weights, subset, k):
    dists = dtoti(weights, subset)
    dists.sort(key=lambda x: x[1])
    if k == 1:
        return dists[0][0].cindex
    counter = {}
    for timg, dist in dists:
        try:
            counter[timg.cindex] += 1
        except:
            counter[timg.cindex] = 1
        if counter[timg.cindex] == k:
            return timg.cindex
    return max(counter, key=lambda x: counter[x])