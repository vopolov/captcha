from itertools import groupby

import numpy as np


def naive_ctc_decode(array):
    array = array.argmax(axis=1)
    array = [k for k, g in groupby(array)]
    array = np.array([a for a in array if a != 0])
    return array


if __name__ == '__main__':
    test = np.random.rand(81, 10)
    test = naive_ctc_decode(test)
    print(test)
