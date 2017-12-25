from math import isclose
from prep import *
from afr.pathreset import pathreset
from afr.cmc import *


@pytest.fixture(scope='module')
def dfo_dim1(originR2b):
    # dists from originR2b
    return dtocm(originR2b, R2b, 1)

@pytest.fixture(scope='module')
def dfo_dim2(originR2b):
    return dtocm(originR2b, R2b, 0)


#####################################
## Tests
#####################################


def test_dtocm(dfo_dim1, dfo_dim2):
    assert(dfo_dim1 == [8/3, 7/3, 9])
    for i, n in enumerate([np.sqrt(80)/3, np.sqrt(65)/3, np.sqrt(181)]):
        assert(isclose(dfo_dim2[i], n))