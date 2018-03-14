from prep import *
from afr.pca import *


####################################################
## Fixtures
####################################################


# training values
rmatrix = [
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
]
# let L = rmatrix * transpose(rmatrix)

# the eigenvalues of L are 2 and 0, each with multiplicity 3
# numpy will calculate the 3 eigenvectors corresponding to eigenvalue 2 to be:
#   [-sqrt(2)] + [0]*9
#   [0, -sqrt(2)] + [0]*8
#   [0, 0, -sqrt(2)] + [0]*7

# after normalizing, the eigenvalues of L will be 2*sqrt(2) and 0
# the eigenvectors corresponding to 2*sqrt(2) are then unit vectors


# function results
mean, eigvs, eigfs = build_reqs(rmatrix)
neigvs, neigfs = eigvs.copy(), eigfs.copy()
normalize(neigvs, neigfs)


####################################################
## Tests
####################################################


def test_build_reqs():
    assert(mean.astype(np.int).tolist() == [1]*10)
    assert(eigvs.astype(np.int).tolist() == [2, 2, 2, 0, 0, 0])
    assert(np.linalg.matrix_rank(eigfs) == 3)


def test_normalize():
    trt = 2*np.sqrt(2)
    assert(np.allclose(neigvs, [trt, trt, trt, 0, 0, 0]))
    
    for j in range(neigfs.shape[1]):
        norm = np.linalg.norm(neigfs[:,j])
        if norm > 0:
            assert(norm == 1)


def test_pca():
    m, v, f = pca(rmatrix)
    assert(np.array_equal(mean, m))
    assert(np.array_equal(neigvs, v))
    assert(np.array_equal(neigfs, f))