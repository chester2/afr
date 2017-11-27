# Functions for determining eigenface class weights and working with weights.


import numpy as np
from ..prereq.consts import *
from ..prereq.imio import *
from ..prereq.path import *
from ..prereq.dbio import *


#####################################
## Helper Functions
#####################################


def get_classid(setname, i):
    # returns the filename string pattern that identifies a face's class
    return f'{SETS[setname][PFX]}{i}{SETS[setname][SFX]}'


def build_cdb(setname):
    # returns a 3D list (class, face, pixel intensity)
    pathset(setname)
    cdb = []
    files = ls()
    for i in range(SETS[setname][INIT], SETS[setname][END] + 1):
        cdb.append([
            imread(filename)
            for filename in files
            if get_classid(setname, i) in filename
        ])
    return cdb


def build_cweights(class_list, mean, eigfs):
    # returns a column matrix of weights of class_list's faces
    # class_list is a 2D list of class faces (row vectors)
    cshifted = np.transpose(np.array(class_list, dtype=np.float64) - mean)
    return np.transpose(eigfs).dot(cshifted)


def build_fweights(face, mean, eigfs):
    # returns a vector of face's weights
    # face is a list (row vector)
    fshifted = np.array(face, dtype=np.float64) - mean
    return np.transpose(eigfs).dot(fshifted)


#####################################
## Main Functions
#####################################


@pathreset
def eigf(setname):
    mean = readmean(setname)
    eigfs = readeigfs(setname)
    cdb = build_cdb(setname)
    cmeanweights = []
    for i, c in enumerate(cdb):
        cweights = build_cweights(c, mean, eigfs)
        writeclass(setname, i, cweights)
        cmeanweights.append(np.mean(cweights, 1))
    temp = np.transpose(np.array(cmeanweights, dtype=np.float64))
    writecmeans(setname, temp)


def f_to_weights(setname, filepath):
    # import a face image and convert to eigenface weights
    # filepath is absolute
    mean = readmean(setname)
    eigfs = readeigfs(setname)
    face = imread(filepath)
    return build_fweights(face, mean, eigfs)