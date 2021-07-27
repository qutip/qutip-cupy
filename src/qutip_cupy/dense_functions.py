# Contains functions for dense_cupy, this are the same functions that
# are defined ouside the dense.pyx file

import cupy as cp

from .dense import CuPyDense


def tidyup_dense(matrix, tol, inplace=True):
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
    # @TODO: whnen qutip allows it we should remove this call to item()
    # as it takes a time penalty commmunicating data from GPU to CPU.
    return cp.trace(cp_arr._cp).item()
