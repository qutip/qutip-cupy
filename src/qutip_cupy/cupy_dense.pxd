#cython: language_level=3

cimport numpy as cnp

from qutip.core.data cimport base



cdef class Dense(data.Data):
    cdef object _cp
    