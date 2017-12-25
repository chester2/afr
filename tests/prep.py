import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from afr import Imset
from afr.eigf import build_cmeans


ROOT = os.getcwd()

R2b = Imset(
    name='R2b',
    width=2,
    height=1,
    ipfx='',
    isfx='',
    ifirst=0,
    ifinal=2,
    dir_to_ims=os.path.join(ROOT),
    dir_to_npy=os.path.join(ROOT))


# Everything below is for test_knn and test_cmc

@pytest.fixture(scope='session')
def originR2b():
    return np.array([0, 0], dtype=np.float64)

@pytest.fixture(scope='session')
def cwR2b():
    class0 = np.array([
        [1, 0],
        [-4, -4],
        [-5, 0]], dtype=np.float64).transpose()
    class1 = np.array([
        [0, 2],
        [1, 2],
        [-8, 0]], dtype=np.float64).transpose()
    class2 = np.array([
        [10, 9],
        [6, 10],
        [11, 11]], dtype=np.float64).transpose()
    return [class0, class1, class2]

@pytest.fixture(scope='session')
def cwR2b_setup_teardown(cwR2b):
    for i, cw in enumerate(cwR2b):
        R2b.writecweights(i, cw)
    R2b.writecmeans(build_cmeans(cwR2b))
    yield
    R2b.clearnpy()
def test_cwR2b_setup_teardown(cwR2b_setup_teardown):
    pass