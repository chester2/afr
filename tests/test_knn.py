from math import isclose
from prep import *
from afr.pathreset import pathreset
from afr.knn import *


@pytest.fixture(scope='module')
def dfo_dim1(originR2b):
    # dists from originR2b
    return dtoa(originR2b, R2b, 1)

@pytest.fixture(scope='module')
def dfo_dim2(originR2b):
    return dtoa(originR2b, R2b, 0)


#####################################
## Tests
#####################################


def test_dtoc():
    fw = np.array([1, 2, 3], dtype=np.float64)
    cw = np.array([
        [1, 4],
        [-1, 9],
        [-5, 0]], dtype=np.float64)
    dtoc_dim1 = dtoc(fw, cw, 0, 1)
    dtoc_dim2 = dtoc(fw, cw, 0, 2)
    dtoc_dim3 = dtoc(fw, cw, 0, 0)
    
    assert(dtoc_dim1[0][1] == 0)
    assert(dtoc_dim1[1][1] == 3)
    
    assert(dtoc_dim2[0][1] == 3)
    assert(isclose(dtoc_dim2[1][1], np.sqrt(58)))
    
    assert(isclose(dtoc_dim3[0][1], np.sqrt(73)))
    assert(isclose(dtoc_dim3[1][1], np.sqrt(67)))
    assert(dtoc_dim3 == dtoc(fw, cw, 0, 3))


def test_dtoa(dfo_dim1, dfo_dim2):
    assert(dfo_dim1 == [
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 4),
        (0, 5),
        (2, 6),
        (1, 8),
        (2, 10),
        (2, 11)])
    assert(dfo_dim2 == [
        (0, 1),
        (1, 2),
        (1, np.sqrt(5)),
        (0, 5),
        (0, np.sqrt(32)),
        (1, 8),
        (2, np.sqrt(136)),
        (2, np.sqrt(181)),
        (2, np.sqrt(242))])


def test_identify(dfo_dim1, dfo_dim2):
    assert(identify(dfo_dim1, 1) == 1)
    assert(identify(dfo_dim1, 2) == 1)
    assert(identify(dfo_dim1, 3) == 0)
    
    assert(identify(dfo_dim2, 1) == 0)
    assert(identify(dfo_dim2, 2) == 1)
    assert(identify(dfo_dim2, 3) == 0)