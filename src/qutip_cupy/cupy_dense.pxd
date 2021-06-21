#cython: language_level=3

cimport numpy as cnp

from qutip.core.data cimport base



cdef class CuPyDense(base.Data):
    cpdef object _cp
    