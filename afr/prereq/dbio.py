# Functions for reading and writing npy files.


import numpy as np
from ..prereq.consts import *
from ..prereq.path import *


def readmean(setname):
    pathdb()
    return np.load(f'{setname}_mean.npy')
def readeigvs(setname):
    pathdb()
    return np.load(f'{setname}_eigvs.npy')
def readeigfs(setname):
    pathdb()
    return np.load(f'{setname}_eigfs.npy')
def readclass(setname, i):
    pathdb()
    return np.load(f'{setname}_class{i}.npy')
def readcmeans(setname):
    pathdb()
    return np.load(f'{setname}_cmeans.npy')


def writemean(setname, mean):
    pathdb()
    np.save(f'{setname}_mean', mean)
def writeeigvs(setname, l):
    pathdb()
    np.save(f'{setname}_eigvs', l)
def writeeigfs(setname, u):
    pathdb()
    np.save(f'{setname}_eigfs', u)
def writeclass(setname, i, cweights):
    pathdb()
    np.save(f'{setname}_class{i}', cweights)
def writecmeans(setname, cmeanweights):
    pathdb()
    np.save(f'{setname}_cmeans', cmeanweights)