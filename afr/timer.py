import functools
from time import time

def timer(do):
    def wrapper(f):
        @functools.wraps(f)
        def wrapper2(*args, **kwargs):
            if do:
                t0 = time()
                x = f(*args, **kwargs)
                t1 = time()
                print(f, '\n    ', t1 - t0)
                return x
            return f(*args, **kwargs)
        return wrapper2
    return wrapper