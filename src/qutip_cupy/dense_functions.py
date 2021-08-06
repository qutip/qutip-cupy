"""Contains specialization functions for dense_cupy. These are the functions that
 are defined outside of qutip/core/data/dense.pyx."""

import cupy as cp

from .dense import CuPyDense


def tidyup_dense(matrix, tol, inplace=True):
    return matrix


def reshape_cupydense(cp_arr, n_rows_out, n_cols_out):

    return CuPyDense._raw_cupy_constructor(
        cp.reshape(cp_arr._cp, (n_rows_out, n_cols_out))
    )


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


def frobenius_cupydense(cp_arr):
    # TODO: Expose CUBLAS' dznrm2 (like QuTiP does) and test if it is faster
    return cp.linalg.norm(cp_arr._cp).item()


def l2_cupydense(cp_arr):
    if cp_arr.shape[0] != 1 and cp_arr.shape[1] != 1:
        raise ValueError("L2 norm is only defined on vectors")
    return frobenius_cupydense(cp_arr)


def max_cupydense(cp_arr):
    return cp.max(cp.abs(cp_arr._cp)).item()


def one_cupydense(cp_arr):
    return cp.linalg.norm(cp_arr._cp, ord=1).item()
  
 
def pow_cupydense(cp_arr, n):
    if cp_arr.shape[0] != cp_arr.shape[1]:
        raise ValueError("matrix power only works with square matrices")

    out_arr = cp.linalg.matrix_power(cp_arr._cp, n)

    return CuPyDense._raw_cupy_constructor(out_arr)


def project_cupydense(state):
    """
    Calculate the projection |state><state|.  The shape of `state` will be used
    to determine if it has been supplied as a ket or a bra.  The result of this
    function will be identical is passed `state` or `adjoint(state)`.
    """

    if state.shape[1] == 1:
        return CuPyDense._raw_cupy_constructor(cp.outer(state._cp, state.adjoint()._cp))
    elif state.shape[0] == 1:
        return CuPyDense._raw_cupy_constructor(cp.outer(state.adjoint()._cp, state._cp))
    else:
        raise ValueError("state must be a ket or a bra.")
