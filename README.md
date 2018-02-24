# Automated Face Recognition

Python implementation of the [eigenface method](https://www.cs.ucsb.edu/~mturk/Papers/jcn.pdf) for facial recognition.

Images are assumed to cluster around a submanifold of a high-dimensional Euclidean space. Approximate that manifold by a collection of lower-dimensional linear subspaces, applying the eigenface method to each linear subspace.



### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [ImgSet Reference](#reference)



<br>
<h2 id="installation">Installation</h2>

### Requirements

- Python 3.6+
- [Pillow](https://python-pillow.org)
- [NumPy](http://www.numpy.org)
- [SciPy](https://www.scipy.org/scipylib/index.html)
- [scikit-learn](http://scikit-learn.org)



### With Git

Open the Git CLI and run:

```
pip install git+https://github.com/chester2/afr.git
```



### Without Git

Extract the entire archive to a directory, cd there from a command line, then run:
```
python setup.py install
```



<br>
<h2 id="usage">Usage</h2>

The `ImgSet` and `Img` classes make up the core of AFR. `ImgSet` is a training set metadata container, and `Img` is an image metadata container. All important operations are methods of `ImgSet`, which may or may not take an `Img` object as an argument.

### Preparation of Training Images

- All images in a training set must be of identical dimensions.
- In the filename of any training image must be a string pattern of the form `ipfx + str(i) + isfx`, where `i` is the class index of that training image. `i` may be preceeded by any number of zeros.
- The directory containing the training images must contain no other files or subdirectories.
- A "to-be-identified image" processed through a particular `ImgSet` object must have identical dimensions to that set's training images.

### Instantiating an ImgSet Object

Full class definition:

    class afr.ImgSet(
        name,               # used to tag filenames of cache files
        width,              # pixel width
        height,             # pixel height
        ipfx,               # class index prefix
        isfx,               # class index suffix
        nofss,              # number of linear subsets to use
        timg_dir,           # abs path to directory containing training images
        cache_dir,          # abs path to where cache files should be stored
        class_names=None,   # list/dict mapping class indices to class names
        refresh_db=False    # remakes cache files if True
    )

Sample instantiation for the [yale face database](http://vismod.media.mit.edu/vismod/classes/mas622-00/datasets/):

    >>> import os
    >>> from afr import ImgSet
    >>> yf = ImgSet(
    >>>     "yalefaces",
    >>>     320,
    >>>     243,
    >>>     "subject",
    >>>     ".",
    >>>     7,
    >>>     os.path.abspath("c:/yalefaces/"),
    >>>     os.path.abspath("c:/db/")
    >>> )

On instantiating an `ImgSet` object, cache files are automatically created if they do not exist for the current `nofss` value. On setting a new `nofss` value, cache files will be automatically created if they do not exist for the new value.

### Instantiating an Img Object

The only argument that an `Img` object takes on instantiation is the image's absolute filepath.

    >>> import os
    >>> from afr import Img
    >>> im = Img(os.path.abspath("c:/a_face.jpg"))

### Classifying an Image

Call an `ImgSet` object's `cmc` or `nn` method on an `Img` object.

    >>> yf.cmc(im)
    (3, '')
    >>> yf.nn(im)
    (3, '')



<br>
<h2 id="reference">ImgSet Reference</h2>

### Attributes

<code>imgset.<b>rmatrix</b></code><br>
A 2D list. Each inner list represents a training image and contains its pixel intensities (integers from 0-255).
<br>
<br>

<code>imgset.<b>ss_sizes</b></code><br>
A list of integers. `imgset.ss_sizes[i]` is the number of training images that make up subset *i*, where *i* ranges from 0 to `imgset.nofss - 1`.
<br>
<br>



### Methods

<code>imgset.<b>cmc</b>(<i>img</i>)</code><br>
<code>imgset.<b>nn</b>(<i>img</i>)</code><br>
Classify an image and return a 2-tuple.

The first element is the matching class index.

The second element is the class name corresponding to that index if `imgset.class_names` exists and is valid. Otherwise, the second element is the empty string.

The difference between these methods is that `cmc` looks for the closest class mean training image while `nn` simply looks for the closest training image.
<br>
<br>

<code>imgset.<b>clear_cache</b>()</code><br>
Deletes all cache files associated with `imgset`.

To continue using `imgset` without reinstantiating it, set its `nofss` attribute.
<br>
<br>

<code>imgset.<b>ss_by_dist</b>(<i>img</i>)</code><br>
Returns a list of subset indices, ordered by closest to farthest from `img`. Distances are measured between `img` and subset means.
<br>
<br>

<code>imgset.<b>itow</b>(<i>img, ssindex</i>)</code><br>
Returns a NumPy array of `img`'s eigenface weights when considering the subset indexed by `ssindex`.

Array element *i* is the weight corresponding to eigenface *i*, where eigenface *i* has the *i+1*<sup>th</sup> largest eigenvalue of all eigenfaces in that subset.
<br>
<br>

<code>imgset.<b>rmk_img</b>(<i>img, ssindex, rmk_dir</i>)</code><br>
Reconstructs an image using the eigenfaces of the desired subset.
<br>
<br>

<code>imgset.<b>rmk_mean</b>(<i>ssindex, rmk_dir</i>)</code><br>
Exports the mean of the desired subset as an image file.
<br>
<br>