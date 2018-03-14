# Algorithm accuracy evaluation.

# How to use:

# Import this module.
# Define the ImgSet object.
# Define a dict "tbcmap" with:
#   key = absolute filepath of a to-be-classified (TBC) image
#   value = index of the class this image should belong to
# Run eval.setup(tbcmap), which returns a dict "tbcimgs" with:
#   key = Img object representing a TBC image
#   value = index of the class this image should belong to
# Run eval.cmc(imgset, tbcimgs) or eval.knn(imgset, tbcimgs[, k]) to evaluate performance. These functions both return a list of Img objects which were classified incorrectly. For each incorrectly classified image, logs relevant distance information to log files in imgset.cache_dir.


from prep import *
from afr import ImgSet, Img


def setup(tbcmap):
    return {
        Img(k):v
        for k, v in tbcmap.items()
    }


def cmc(imgset, tbcimgs):
    errors = []
    for img in tbcimgs:
        match = imgset.cmc(img)[0]
        if tbcimgs[img] != match:
            errors.append(img)
    if errors:
        imgset.log_imgs(errors, 'cm')
    return errors


def knn(imgset, tbcimgs, k=2):
    errors = []
    for img in tbcimgs:
        match = imgset.knn(img, k)[0]
        if tbcimgs[img] != match:
            errors.append(img)
    if errors:
        imgset.log_imgs(errors, 'ti')
    return errors