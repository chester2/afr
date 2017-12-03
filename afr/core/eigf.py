# Functions for determining eigenface class weights and working with weights.


import numpy as np
from ..prereq.consts import *
from ..prereq.imio import *
from ..prereq.path import *
from ..prereq.dbio import *


#####################################
## Helper Functions
#####################################


def get_classid(i, setname):
    # returns the filename string pattern that identifies a face's class
    return f'{SETS[setname][PFX]}{i}{SETS[setname][SFX]}'


def build_face_classes(setname):
    # returns a 3D list (class, face, pixel intensity)
    pathset(setname)
    face_classes = []
    files = ls()
    for i in range(SETS[setname][INIT], SETS[setname][END] + 1):
        face_classes.append([
            imread(filename)
            for filename in files
            if get_classid(i, setname) in filename
        ])
    return face_classes


def build_cweights(face_class, mean, eigfs):
    # returns a column matrix of weights of face_class's faces
    # class_list is a 2D list of class faces (row vectors)
    mean_shifted = np.transpose(np.array(face_class, dtype=np.float64) - mean)
    return np.transpose(eigfs).dot(mean_shifted)


def build_fweights(face, mean, eigfs):
    # returns a vector of face's weights
    # face is a list (row vector)
    mean_shifted = np.array(face, dtype=np.float64) - mean
    return np.transpose(eigfs).dot(mean_shifted)


#####################################
## Main Functions
#####################################


@pathreset
def eigf(setname):
    mean = readmean(setname)
    eigfs = readeigfs(setname)
    face_classes = build_face_classes(setname)
    cmeanweights = []
    for i, face_class in enumerate(face_classes):
        cweights = build_cweights(face_class, mean, eigfs)
        writeclass(setname, i, cweights)
        cmeanweights.append(np.mean(cweights, 1))
    temp = np.transpose(np.array(cmeanweights, dtype=np.float64))
    writecmeans(setname, temp)


def f_to_weights(filepath, setname):
    # import a face image and convert to eigenface weights
    # filepath is absolute
    mean = readmean(setname)
    eigfs = readeigfs(setname)
    face = imread(filepath)
    return build_fweights(face, mean, eigfs)