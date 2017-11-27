# Aliases for os module cwd manipulation functions, plus other path-related functions.

import functools
import os
from ..prereq.consts import *


ls = os.listdir

def pathset(setname):
    # go to the directory where setname's training images are located
    os.chdir(SETS[setname][PATH])
def pathdb():
    os.chdir(PATHDB)
def pathexports():
    os.chdir(PATHEXPORTS)


def pathreset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        root = os.getcwd()
        x = func(*args, **kwargs)
        os.chdir(root)
        return x
    return wrapper


@pathreset
def listtbi():
    os.chdir(PATHTBI)
    return ls()