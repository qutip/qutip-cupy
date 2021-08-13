import cupy as cp


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
        return _expect_dense_ket(op, state)
    return _expect_dense_dense_dm(op, state)


def _expect_dense_ket(op, state):
    _check_shape_ket(op, state)
    return cp.vdot(state._cp, op._cp @ state._cp).item()


def _check_shape_dm(op, state):
    if (
        op.shape[1] != state.shape[0]  # Matrix multiplication
        or state.shape[0] != state.shape[1]  # State is square
        or op.shape[0] != op.shape[1]  # Op is square
    ):
        raise ValueError(
            "incorrect input shapes " + str(op.shape) + " and " + str(state.shape)
        )


_expect_dense_kernel = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>
    extern "C" __global__
    void expect_dens(const complex<double>* op,const complex<double>* dm,
                                        const int size, complex<double>* y) {
        for (unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x; tidx < size;
                                                      tidx += gridDim.x * blockDim.x) {
            for(unsigned int j= 0; j<size; j++){
                    y[tidx] += op[j*size+tidx] * dm[tidx*size+j];
                };
        };
    }""",
    "expect_dens",
)


def _expect_dense_dense_dm(op, state):
    _check_shape_dm(op, state)
    size = op.shape[0]
    out = cp.zeros((size,), dtype=cp.complex128)
    # TODO: check if there is a better way to set thread dim and block dim
    block_size = 64
    grid_size = (size + block_size - 1) // block_size
    _expect_dense_kernel((grid_size,), (block_size,), (op._cp, state._cp, size, out))
    return out.sum().item()


def _check_shape_ket(op, state):
    if (
        op.shape[1] != state.shape[0]  # Matrix multiplication
        or state.shape[1] != 1  # State is ket
        or op.shape[0] != op.shape[1]  # op must be square matrix
    ):
        raise ValueError(
            "incorrect input shapes " + str(op.shape) + " and " + str(state.shape)
        )
