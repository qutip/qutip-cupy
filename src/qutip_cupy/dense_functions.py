# Contains functions for dense_cupy, this are the same functions that
# are defined ouside the dense.pyx file

import cupy as cp
from cupy import cublas

from . import CuPyDense


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
        return _expect_dense_ket(op, state)
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
        return _expect_dense_ket(op, state)
    return _expect_dense_dense_dm(op, state)


def _expect_dense_ket_naive(op, state):

    return state.adjoint() @ (op @ state)


def _expect_dense_ket_naive2(op, state):

    return cp.vdot(state, op @ state)


def _expect_dense_ket_cublas(op, state):

    out = cp.zeros((1, 1), cp.complex128)

    cublas.gemm("H", "N", state, op @ state, out=out)

    return out.item()


def _expect_dense_ket_kernel(op, state):
    # _check_shape_ket(op, state)
    # cdef double complex out=0, sum
    # cdef size_t row, col, op_row_stride, op_col_stride
    # op_row_stride = 1 if op.fortran else op.shape[1]
    # op_col_stride = op.shape[0] if op.fortran else 1

    # for row in range(op.shape[0]):
    #     sum = 0
    #     for col in range(op.shape[0]):
    #         sum += (op.data[row * op_row_stride + col * op_col_stride] *
    #                 state.data[col])
    #     out += sum * conj(state.data[row])
    # return out

    # This is the more complex implementation
    # per device optimization of kernel parameters has to be studied

    complex_kernel = cp.RawKernel(
        r"""
    #include <cupy/complex.cuh>
    extern "C" __global__
    void expect(const complex<float>* d_M, const complex<float>* d_N,
                complex<float>* d_P, int Width) {

        __ shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
        __ shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

        int bx = blockIdx.x; int by = blockIdx.y;
        int tx = threadIdx.x; int ty = threadIdx.y;
        // Identify the row and column of the d_P element to work on
        int Row = by * TILE_WIDTH + ty;
       int Col = bx * TILE_WIDTH + tx;
               float Pvalue = 0;
        // Loop over the d_M and d_N tiles required to compute the d_P element

        for (int m = 0; m < Width/TILE_WIDTH; ++m) {
        // Coolaborative loading of d_M and d_N tiles into shared memory
               Mds[tx][ty] = d_M [Row*Width + m*TILE_WIDTH+tx ];

        Nds[tx][ty] = d_N [(m*TILE_WIDTH+ty )*Width + Col];

        syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
        Pvalue += Mds[tx][k] * Nds[k][ty];
        __ synchthreads();
        }
                }
        d_P [Row*Width+Col] = Pvalue;


     }
    return out
    """,
        "my_func",
    )


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


def stack_cupydense(cp_arr, inplace=False):

    pass


def unstack_cupydense(cp_arr, idxint, inplace=False):

    pass


# reshape
# stack
# kron

# dot

# eigval
# pow
# expm
