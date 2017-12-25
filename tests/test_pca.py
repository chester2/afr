from math import isclose
from prep import *
from afr.pca import *


R3 = Imset(
    name='R3',
    width=10,
    height=1,
    ipfx='',
    isfx='',
    ifirst=0,
    ifinal=5,
    dir_to_ims=os.path.join(ROOT, 'R3'),
    dir_to_npy=os.path.join(ROOT))

@pytest.fixture(scope='module')
def teardown():
    yield
    R3.clearnpy()
def test_run_teardown(teardown):
    pass

@pytest.fixture(scope='module')
def rmatrix():
    return import_ims(R3.dir_to_ims)

@pytest.fixture()
def reqs(rmatrix):
    return build_reqs(rmatrix)

@pytest.fixture()
def normed(reqs):
    eigvs = reqs[1]
    eigfs = reqs[2]
    normalize(eigvs, eigfs)
    return eigvs, eigfs


#####################################
## Tests
#####################################


def test_import_ims(rmatrix):
    x = rmatrix.astype(np.int).tolist()
    for i in range(3):
        v = [1] * 10
        v[i] = 2
        assert v in x
        v[i] = 0
        assert v in x


def test_build_reqs(reqs):
    mean, eigvs, eigfs = reqs
    assert(list(mean.astype(np.int)) == [1]*10)
    assert(list(eigvs.astype(np.int)) == [2]*3 + [0]*3)
    assert(np.linalg.matrix_rank(eigfs) == 3)


def test_normalize(normed):
    eigvs, eigfs = normed
    two_root2 = 2*np.sqrt(2)
    for i in range(3):
        assert(isclose(eigvs[i], two_root2))
    for j in range(eigfs.shape[1]):
        x = np.linalg.norm(eigfs[:,j])
        if x > 0:
            assert(x == 1)


def test_pca(normed):
    pca(R3)
    mean = R3.readmean()
    eigvs = R3.readeigvs()
    eigfs = R3.readeigfs()
    neigvs, neigfs = normed
    assert(list(mean) == [1]*10)
    assert(list(eigvs) == list(neigvs))
    assert(eigfs.tolist() == neigfs.tolist())