from prep import *
from afr.pathreset import pathreset
from afr.eigf import *


R2 = Imset(
    name='R2',
    width=11,
    height=1,
    ipfx='c',
    isfx='_',
    ifirst=0,
    ifinal=2,
    dir_to_ims=os.path.join(ROOT, 'R2'),
    dir_to_npy=os.path.join(ROOT))

@pytest.fixture(scope='module')
def zeromeanR5():
    return np.array([0]*7, dtype=np.float64)

@pytest.fixture(scope='module')
def stdbasisR5():
    return np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.float64)


#####################################
## Tests
#####################################


def test_build_fclasses():
    fcR2 = build_fclasses(R2)
    assert(len(fcR2) == 3)
    assert(len(fcR2[0]) == 4)
    assert(len(fcR2[1]) == 3)
    assert(sorted(fcR2[0]) == [
        [0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [127, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [255, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert(sorted(fcR2[2]) == [
        [119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [127, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [128, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


def test_build_cweights(zeromeanR5, stdbasisR5):
    fclass = [
        [100, 255, 0, 123, 8, 1, 2],
        [242, 100, 248, 0, 1, 3, 4],
        [1, 2, 3, 4, 5, 5, 6]]
    cweights = build_cweights(fclass, zeromeanR5, stdbasisR5).astype(np.int).transpose()
    for i in range(cweights.shape[0]):
        for j in range(5):
            assert(cweights[i,j] == fclass[i][j])


def test_build_cmeans():
    cmeans = R2b.readcmeans()
    assert(cmeans.transpose().tolist() == [
        [-8/3, -4/3],
        [-7/3, 4/3],
        [9, 10]])


def test_build_fweights(zeromeanR5, stdbasisR5):
    face = [100, 255, 0, 123, 8, 1, 2]
    fweights = build_fweights(face, zeromeanR5, stdbasisR5).astype(np.int)
    for i in range(5):
        assert(fweights[i] == face[i])