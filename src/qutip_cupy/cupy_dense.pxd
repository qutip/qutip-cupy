#cython: language_level=3

cimport cython
cimport numpy as cnp

from qutip.core.data cimport base



cdef class CuPyDense(base.Data):
    cdef object cpar
    