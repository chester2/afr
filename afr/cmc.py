# Class mean (average) classification.


import numpy as np


def get_cmeans(subset):
    # returns a dictionary mapping each class index to a np array containing the average weights of that class
    cmeans = {}
    for timg in subset:
        try:
            cmeans[timg.cindex].append(timg.weights)
        except:
            cmeans[timg.cindex] = [timg.weights]
    for cindex in cmeans:
        cmeans[cindex] = np.mean(cmeans[cindex], 0, dtype=np.float64)
    return cmeans


def dtocm(weights, cmeans):
    # distances to class means
    dists = {}
    for cindex in cmeans:
        dists[cindex] = np.linalg.norm(weights - cmeans[cindex])
    return dists


def cmc(weights, subset):
    cmeans = get_cmeans(subset)
    dists = dtocm(weights, cmeans)
    return min(dists, key=lambda x: dists[x])