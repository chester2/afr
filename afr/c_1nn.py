# 1-nearest neighbor classification.


import numpy as np


def dtoti(weights, subset):
    # distances to training images
    dists = {}
    for timg in subset:
        dists[timg] = np.linalg.norm(weights - timg.weights)
    return dists


def nn(weights, subset):
    dists = dtoti(weights, subset)
    return min(dists, key=lambda x: dists[x]).cindex