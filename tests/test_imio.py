from prep import *
from afr.imio import *


####################################################
## Fixtures
####################################################


# training values
pixels = [1, 2, 3, 4]


# function results
stdpixels = standardize(pixels)
unstdpixels = unstandardize(stdpixels)


####################################################
## Tests
####################################################


def test_standardize():
    assert(np.allclose(stdpixels, [
        -1.3416407864998738,
        -0.44721359549995793,
        0.44721359549995793,
        1.3416407864998738
    ]))


def test_unstandardize():
    assert(np.allclose(
        unstdpixels,
        [0, 85, 170, 255]
    ))