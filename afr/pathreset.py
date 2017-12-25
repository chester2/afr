# Path reset decorator.

import functools
import os

def pathreset(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        root = os.getcwd()
        x = f(*args, **kwargs)
        os.chdir(root)
        return x
    return wrapper