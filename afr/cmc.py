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
    return {
        cindex: np.mean(cmeans[cindex], 0, dtype=np.float64)
        for cindex in cmeans
    }


def dtocm(weights, cmeans):
    # distances to class means
    return {
        cindex: np.linalg.norm(weights - cmeans[cindex])
        for cindex in cmeans
    }


def cmc(weights, subset):
    cmeans = get_cmeans(subset)
    dists = dtocm(weights, cmeans)
    return min(dists, key=lambda x: dists[x])