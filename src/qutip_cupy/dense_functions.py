# Contains functions for dense_cupy, this are the same functions that
# are defined ouside the dense.pyx file

import cupy as cp

from .dense import CuPyDense


def tidyup_dense(matrix, tol, inplace=True):
    # cdef Dense out = matrix if inplace else matrix.copy()
    # cdef double complex value
    # cdef size_t ptr
    # for ptr in range(matrix.shape[0] * matrix.shape[1]):
    #     value = matrix.data[ptr]
    #     if fabs(value.real) < tol:
    #         matrix.data[ptr].real = 0
    #     if fabs(value.imag) < tol:
    #         matrix.data[ptr].imag = 0
    # return out
    pass


def reshape_cupydense(cp_arr, n_rows_out, n_cols_out):

    return CuPyDense(cp_arr, (n_rows_out, n_cols_out))


def _check_square_matrix(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "".join(["matrix shape ", str(matrix.shape), " is not square."])
        )


def trace_cupydense(cp_arr):
    _check_square_matrix(cp_arr)
    return cp.trace(cp_arr._cp).item()
