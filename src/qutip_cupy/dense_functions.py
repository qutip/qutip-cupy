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


def _check_shape_inner(left, right):
    if (left.shape[0] != 1 and left.shape[1] != 1) or right.shape[1] != 1:
        raise ValueError(
            "incompatible matrix shapes " + str(left.shape) + " and " + str(right.shape)
        )


def inner_cupydense(left, right, scalar_is_ket=False):

    _check_shape_inner(left, right)

    if left.shape[0] == left.shape[1] == right.shape[1] == 1:
        if not cp.all(left._cp == 0) and not cp.all(right._cp == 0):
            print(left._cp)
            print(right._cp)
            return (
                (cp.conj(left._cp[0, 0]) * right._cp[0, 0]).item()
                if scalar_is_ket
                else (left._cp[0, 0] * right._cp[0, 0]).item()
            )
        return 0.0j

    if left.shape[0] == 1:
        # TODO check if this runs faster using qutip cublas methods
        # which allow to pass a flag to take conj
        return cp.vdot(cp.conj(left._cp), right._cp).item()

    # TODO:remove the final .item when
    # possible as it forces CPU GPU  transfers.

    return cp.vdot(left._cp, right._cp).item()
