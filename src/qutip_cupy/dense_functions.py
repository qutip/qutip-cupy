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


hermdiff_kernel = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>
    extern "C" __global__
    void hermdiff(const complex<double>* x1,const int size,const double tol, bool* y) {
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        int tidy = blockDim.y * blockIdx.y + threadIdx.y;
        if((tidx < size) & (tidy < size)){
            y[tidx+size*tidy] = norm(x1[tidx*size+tidy]
                                    - conj(x1[tidy*size+tidx])) < tol;
        };
    }""",
    "hermdiff",
)


def isherm_cupydense(cp_arr, tol):
    if cp_arr.shape[0] != cp_arr.shape[1]:
        return False
    size = cp_arr.shape[0]
    diff = cp.empty((size, size), dtype=cp.bool_)
    # TODO: check if there is a better way to set thread dim and block dim
    block_size = 32
    grid_size = (size + block_size - 1) // block_size
    hermdiff_kernel(
        (grid_size, grid_size), (block_size, block_size), (cp_arr._cp, size, tol, diff)
    )
    return diff.all().item()
