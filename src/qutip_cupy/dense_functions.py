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


_hermdiff_kernel = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>
    extern "C" __global__
    void hermdiff(const complex<double>* x1,const int size,const double tol, bool* y) {
        for (unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x; tidx < size;
                                                    tidx += gridDim.x * blockDim.x) {
        for (unsigned int tidy = blockDim.y * blockIdx.y + threadIdx.y;
                                        tidy <= tidx; tidy += gridDim.y * blockDim.y) {

            y[tidx+size*tidy] = norm(x1[tidx*size+tidy]
                                    - conj(x1[tidy*size+tidx])) < tol;
        };
        };
    }""",
    "hermdiff",
)


_hermdiff_kernel_half = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>
    extern "C" __global__
    void hermdiff_half(const complex<double>* x1,const int size,const double tol, bool* y){
        for (unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x; tidx < size/2;
                                                    tidx += gridDim.x * blockDim.x) {

            for(unsigned int j= size - tidx; j<size; j++){
                    y[tidx] =  y[tidx] && (norm(x1[j*size+tidx]
                                        - conj(x1[tidx*size+j])) < tol);
                };

            for(unsigned int k= 0; k<=tidx; k++){
                    y[tidx] =  y[tidx] && (norm(x1[k*size+tidx]
                                        - conj(x1[tidx*size+k])) < tol);
                };

        };
    }""",
    "hermdiff_half",
)


def isherm_cupydense(cp_arr, tol):
    size = cp_arr.shape[0]
    diff = cp.ones((size // 2,), dtype=cp.bool_)
    # TODO: check if there is a better way to set thread dim and block dim
    block_size = 32
    grid_size = (size // 2 + block_size - 1) // block_size
    _hermdiff_kernel_half((grid_size,), (block_size,), (cp_arr, size, tol, diff))
    return diff.all()


def kron_cupydense(cp_arr1, cp_arr2):
    return CuPyDense._raw_cupy_constructor(cp.kron(cp_arr1._cp, cp_arr2._cp))


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
