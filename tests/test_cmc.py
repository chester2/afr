from prep import *
from afr.cmc import *
from fixtures_cmc_knn import *


def test_get_cmeans():
    cmeans = get_cmeans(subset)
    for i in range(4):
        mean = np.mean(c[i], 0)
        assert(np.allclose(cmeans[i], mean))


def test_cmc():
    match = cmc(weights_cmc, subset)
    assert(match == 3)