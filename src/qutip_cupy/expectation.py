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


def expect_cupydense(op, state):
    """
    Get the expectation value of the operator `op` over the state `state`.  The
    state can be either a ket or a density matrix.

    The expectation of a state is defined as the operation:
        state.adjoint() @ op @ state
    and of a density matrix:
        tr(op @ state)
    """
    if state.shape[1] == 1:
        return _expect_dense_ket_naive2(op, state)
    return _expect_dense_dense_dm(op, state)


def _expect_dense_ket_naive(op, state):

    return state.adjoint() @ (op @ state)


def _expect_dense_ket_naive2(op, state):
    _check_shape_ket(op, state)
    return cp.vdot(state._cp, op._cp @ state._cp).item()


def _expect_dense_ket_cublas(op, state):
    _check_shape_ket(op, state)
    out = cp.zeros((1, 1), cp.complex128)

    cublas.gemm("H", "N", state._cp, op._cp @ state._cp, out=out)

    return out.item()


def _check_shape_dm(op, state):
    if (
        op.shape[1] != state.shape[0]  # Matrix multiplication
        or state.shape[0] != state.shape[1]  # State is square
        or op.shape[0] != op.shape[1]  # Op is square
    ):
        raise ValueError(
            "incorrect input shapes " + str(op.shape) + " and " + str(state.shape)
        )


def _expect_dense_dense_dm(op, state):
    _check_shape_dm(op, state)

    return cp.sum(op._cp * state._cp.transpose()).item()


def _check_shape_ket(op, state):
    if (
        op.shape[1] != state.shape[0]  # Matrix multiplication
        or state.shape[1] != 1  # State is ket
        or op.shape[0] != op.shape[1]  # op must be square matrix
    ):
        raise ValueError(
            "incorrect input shapes " + str(op.shape) + " and " + str(state.shape)
        )
