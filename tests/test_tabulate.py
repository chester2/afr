from prep import *
from afr.tabulate import tabulate


####################################################
## Fixtures
####################################################


data = [
    ['Value', 'Years Elapsed', 1234],
    [33, 'asdf', 44],
    [1024, '    ', 'hi my name is']
]


####################################################
## Tests
####################################################


def test_tabulate():
    assert(tabulate(data, margin=1) == (
        ' Value Years Elapsed 1234         \n'
        ' 33    asdf          44           \n'
        ' 1024                hi my name is\n'
        '\n'
    ))
    assert(tabulate(data, '--titLe--', 3) == (
        '--titLe--\n'
        '   Value   Years Elapsed   1234         \n'
        '   33      asdf            44           \n'
        '   1024                    hi my name is\n'
        '\n'
    ))