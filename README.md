# Automated Face Recognition

Python implementation of the [eigenface method](https://www.cs.ucsb.edu/~mturk/Papers/jcn.pdf) for facial recognition.



### Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Method Reference](#reference)



<br>
<h2 id="installation">Installation</h2>

### Requirements

- Python 3.6+
- Compatible version of [NumPy](http://www.numpy.org)
- Compatible version of [Pillow](https://python-pillow.org)



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

The `Imset` class is the core of AFR. It is a training set metadata container, and all AFR operations are defined as `Imset` class methods.

Using AFR is simple: instantiate an `Imset` object, then call its `buildnpy()` method to generate all `.npy` file dependencies. At this point, all other methods are available for use.

### Defining an Imset Object

Full class definition:

    class afr.Imset(
        name,           # used to generate .npy filenames
        width,          # pixel width
        height,         # pixel height
        ipfx,           # class index prefix
        isfx,           # class index suffix
        ifirst,         # first class index
        ifinal,         # final class index
        dir_to_ims,     # absolute path to directory storing the images
        dir_to_npy,     # absolute path to where .npy files should be stored
        classid=None    # list of names for individual classes
    )

The `nofc` property is also available, which indicates the number of classes in the training set.

`ipfx`, `isfx`, `ifirst`, and `ifinal` are all with respect to training image filenames. Internally, AFR shifts indices to start counting from 0.

`classid` is optional and maps names to the shifted class indices.

Sample definition for the [yale face database](http://vismod.media.mit.edu/vismod/classes/mas622-00/datasets/):

    >>> import os
    >>> from afr import Imset
    >>> yalefaces = Imset(
    >>>     "yalefaces",
    >>>     320,
    >>>     243,
    >>>     "subject",
    >>>     ".",
    >>>     1,
    >>>     15,
    >>>     os.path.abspath("c:/facerecog/yalefaces/"),
    >>>     os.path.abspath("c:/facerecog/npy/"))

### Remarks

- All images in a training set must be of identical dimensions.
- Any given training image must contain a string pattern (filename identifier) of the form `ipfx + str(i) + isfx` (where `i` is the non-shifted class index) in its filename. `i` may be preceeded by any number of zeros.
- The directory containing the training images must contain no other files or subdirectories.
- A to-be-identified image processed through a particular `Imset` object must have identical dimensions as that set's training images.



<br>
<h2 id="reference">Method Reference</h2>

All vectors and arrays processed through AFR are of type `numpy.float64`.

### Core

<code>Imset.<b>cmc</b>(<i>imfilepath</i>[<i>, dim</i>])</code><br>
Returns the matched shifted class index after performing class-mean classification.

_dim_ is the eigenspace dimension (e.g. `dim=10` indicates to use the 10 largest eigenfaces). By default, the largest eigenspace is used.
<br>
<br>

<code>Imset.<b>knn</b>(<i>imfilepath, k</i>[<i>, dim</i>])</code><br>
Returns the matched shifted class index after performing _k_-nearest neighbors classification.

_dim_ is as described for `Imset.cmc()`.
<br>
<br>



### .npy Manipulation and Access

<code>Imset.<b>buildnpy</b>()</code><br>
Generates all `.npy` file dependencies.
<br>
<br>

<code>Imset.<b>clearnpy</b>()</code><br>
Removes all associated `.npy` files.
<br>
<br>

<code>Imset.<b>readmean</b>()</code><br>
Returns a vector representation of the mean training image.
<br>
<br>

<code>Imset.<b>writemean</b>(<i>mean</i>)</code><br>
Writes a vector representation of the mean training image to disk.
<br>
<br>

<code>Imset.<b>readeigvs</b>()</code><br>
Returns a vector of eigenvalues.
<br>
<br>

<code>Imset.<b>writeeigvs</b>(<i>eigvs</i>)</code><br>
Writes a vector of eigenvalues to disk.
<br>
<br>

<code>Imset.<b>readeigfs</b>()</code><br>
Returns an array of eigenfaces.
<br>
<br>

<code>Imset.<b>writeeigfs</b>(<i>eigfs</i>)</code><br>
Writes an array of eigenfaces to disk.
<br>
<br>

<code>Imset.<b>readcweights</b>(<i>i</i>)</code><br>
Returns a vector of class _i_'s weights.
<br>
<br>

<code>Imset.<b>writecweights</b>(<i>i, cweights</i>)</code><br>
Writes a vector of class _i_'s weights to disk.
<br>
<br>

<code>Imset.<b>readcmeans</b>()</code><br>
Returns an array of class mean weights.
<br>
<br>

<code>Imset.<b>writecmeans</b>(<i>cmeans</i>)</code><br>
Writes an array of class mean weights to disk.
<br>
<br>



### Image Recreation

<code>Imset.<b>rmkim</b>(<i>imfilepath, dir_to_rmk</i>)</code><br>
Saves an image's eigenface reconstruction to _dir_to_rmk_.
<br>
<br>

<code>Imset.<b>rmkmean</b>(<i>dir_to_rmk</i>)</code><br>
Saves the mean training image to _dir_to_rmk_.
<br>
<br>

<code>Imset.<b>rmkeigfs</b>(<i>dir_to_rmk</i>)</code><br>
Saves eigenfaces as viewable images to _dir_to_rmk_.
<br>
<br>

<code>Imset.<b>rmkcmeans</b>(<i>dir_to_rmk</i>)</code><br>
Saves the mean image for each class to _dir_to_rmk_.
<br>
<br>



### Misc.

<code>Imset.<b>fnid</b>(<i>i</i>)</code><br>
Returns the filename identifier for class _i_.
<br>
<br>

<code>Imset.<b>ftow</b>(<i>imfilepath</i>)</code><br>
Returns a vector of a face image's weights.