"""Contains specialization functions for dense_cupy. These are the functions that
 are defined outside of qutip/core/data/dense.pyx."""

from .dense import CuPyDense
import cupy as cp
from cupy import cublas


def expect_dense_naive(op, state):
    """
    Get the expectation value of the operator `op` over the state `state`.  The
    state can be either a ket or a density matrix.

    The expectation of a state is defined as the operation:
        state.adjoint() @ op @ state
    and of a density matrix:
        tr(op @ state)
    """
    if state.shape[1] == 1:
        return _expect_dense_ket_naive(op, state)
    return _expect_dense_dense_dm(op, state)


def expect_dense(op, state):
    """
    Get the expectation value of the operator `op` over the state `state`.  The
    state can be either a ket or a density matrix.

    The expectation of a state is defined as the operation:
        state.adjoint() @ op @ state
    and of a density matrix:
        tr(op @ state)
    """
    if state.shape[1] == 1:
        return _expect_dense_ket_cublas(op, state)
    return _expect_dense_dense_dm(op, state)


def _expect_dense_ket_naive(op, state):

    return state.adjoint() @ (op @ state)


def _expect_dense_ket_naive2(op, state):

    return cp.vdot(state, op @ state)


def _expect_dense_ket_cublas(op, state):

    out = cp.zeros((1, 1), cp.complex128)

    cublas.gemm("H", "N", state, op @ state, out=out)

    return out.item()


def _expect_dense_dense_dm(op, state):
    # _check_shape_dm(op, state)
    # cdef double complex out=0
    # cdef size_t row, col, op_row_stride, op_col_stride
    # cdef size_t state_row_stride, state_col_stride
    # state_row_stride = 1 if state.fortran else state.shape[1]
    # state_col_stride = state.shape[0] if state.fortran else 1
    # op_row_stride = 1 if op.fortran else op.shape[1]
    # op_col_stride = op.shape[0] if op.fortran else 1

    # for row in range(op.shape[0]):
    #     for col in range(op.shape[1]):
    #         out += op.data[row * op_row_stride + col * op_col_stride] * \
    #                state.data[col * state_row_stride + row * state_col_stride]
    # return out
    raise NotImplementedError


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
