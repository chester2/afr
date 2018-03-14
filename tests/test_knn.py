from prep import *
from afr.knn import *
from fixtures_cmc_knn import *


for element in sorted(dtoti(weights_knn, subset), key=lambda x: x[1]):
    print(element)


def test_knn():
    assert(knn(weights_knn, subset, 1) == 1)
    assert(knn(weights_knn, subset, 2) == 2)
    assert(knn(weights_knn, subset, 3) == 2)
    assert(knn(weights_knn, subset, 4) == 1)
    assert(knn(weights_knn, subset, 5) == 1)
    assert(knn(weights_knn, subset, 6) == 1)