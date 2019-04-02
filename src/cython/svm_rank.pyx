# distutils: language = c++
# cython: language_level=3

import sys
import numpy as np

cimport numpy as np
from cpython cimport Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free
from libc.stdio cimport fdopen
from numpy.math cimport INFINITY, NAN
from libc.math cimport sqrt as SQRT

from libc.stdio cimport printf

def test():
    printf("Coucou !")
    return
