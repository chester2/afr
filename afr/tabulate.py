# For ImgSet logging methods.


import numpy as np


def tabulate(list2d, title=None, margin=2):
    data = np.array(list2d, dtype=np.str)
    widths = [
        len(max(data[:,j], key=lambda x: len(x))) + margin
        for j in range(data.shape[1])
    ]
    if title:
        table = title + '\n'
    else:
        table = ''
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            table += (
                (margin*' ' + data[i,j]).ljust(widths[j])
            )
        table += '\n'
    
    return table + '\n'