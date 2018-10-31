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
        return dists[0][0].cindex, dists[0][1]
    # counter format:
    #   {cindex: [count, total dist]}
    counter = {}
    for timg, dist in dists:
        try:
            counter[timg.cindex][0] += 1
            counter[timg.cindex][1] += dist
        except:
            counter[timg.cindex] = [1, dist]
        if counter[timg.cindex][0] == k:
            return timg.cindex, counter[timg.cindex][1]
    c = max(counter, key=lambda x: counter[x])
    return c, counter[c][1]