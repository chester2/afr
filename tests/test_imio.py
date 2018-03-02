from math import isclose
from prep import *
from afr.imio import *


####################################################
## Fixtures
####################################################


# training values
pixels = [1, 2, 3, 4]
mean = 2.5
std = np.sqrt(5)/2


# function results
stdpixels = standardize(pixels)
unstdpixels = unstandardize(stdpixels)


####################################################
## Tests
####################################################


def test_standardize():
    assert(isclose(stdpixels[0], -1.3416407864998738))
    assert(isclose(stdpixels[1], -0.44721359549995793))
    assert(isclose(stdpixels[2], 0.44721359549995793))
    assert(isclose(stdpixels[3], 1.3416407864998738))


def test_unstandardize():
    m = -1.3416407864998738
    diff = 2.6832815729997477
    for i in range(len(pixels)):
        assert(isclose(unstdpixels[i], (stdpixels[i] - m)*255/diff))