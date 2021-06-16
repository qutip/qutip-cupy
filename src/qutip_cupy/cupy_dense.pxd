#cython: language_level=3

cimport numpy as cnp


ctypedef cnp.npy_int32 idxint
cdef int idxint_DTYPE

cdef class Dense:
    cdef public object cpa
    cdef readonly (idxint, idxint) shape
    cpdef object to_array(Dense self)
