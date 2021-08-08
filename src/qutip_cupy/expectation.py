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
