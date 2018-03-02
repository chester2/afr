import numpy as np


class TImgTest:
    def __init__(self, weights, cindex):
        self.weights = weights      # list of numbers
        self.cindex = cindex        # integer
    
    def __str__(self):
        return str(self.cindex)
    
    def __repr__(self):
        return self.__str__()


# training set
c = (
    [
        [3, 3, 3],
        [5, 5, 1]
    ],
    [
        [0, 0, -1],
        [10, 8, 6],
        [9, 4, 4],
        [10, 8, 6],
        [11, 7, 6]
    ],
    [
        [-1, 0, -3],
        [-10, 2, -10],
        [-7, -3, -2]
    ],
    [
        [40, 55, 50],
        [30, 33, 12]
    ]
)
subset =(
        [TImgTest(x, 0) for x in c[0]] +
        [TImgTest(x, 1) for x in c[1]] +
        [TImgTest(x, 2) for x in c[2]] +
        [TImgTest(x, 3) for x in c[3]]
    )


# to-be-classified image
weights_cmc = np.array([45, 60, 30], dtype=np.float64)
weights_knn = np.array([-1, -1, -1], dtype=np.float64)