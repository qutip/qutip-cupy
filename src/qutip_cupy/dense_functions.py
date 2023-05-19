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


def column_stack_cupydense(cp_arr):
    return CuPyDense._raw_cupy_constructor(
        cp.reshape(cp_arr._cp, (-1, 1), order="F")
    )


def column_unstack_cupydense(cp_arr, rows):
    return CuPyDense._raw_cupy_constructor(
        cp.reshape(cp_arr._cp, (rows, -1), order="F")
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


# This kernel checks herm by distributing work among size/2 threads and balanciung out
# the work contiguous threads access contiguous memory location
# (i.e. items in the same row) at least in the first for loop.
_hermdiff_kernel_half = cp.RawKernel(
    r"""
    #include <cupy/complex.cuh>
    extern "C" __global__
    void hermdiff_half(const complex<double>* x1,
                        const int size,const double tol, bool* y){
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
    if cp_arr.shape[0] != cp_arr.shape[1]:
        return False
    size = cp_arr.shape[0]
    diff = cp.ones((size // 2,), dtype=cp.bool_)
    # TODO: check if there is a better way to set thread dim and block dim
    block_size = 256
    grid_size = (size // 2 + block_size - 1) // block_size
    _hermdiff_kernel_half((grid_size,), (block_size,), (cp_arr._cp, size, tol, diff))
    return diff.all().item()


def split_columns_cupydense(cp_arr, copy=True):
    return [CuPyDense(cp_arr._cp[:, k], copy=copy) for k in range(cp_arr.shape[1])]


def _check_shape_inner(left, right):
    if (left.shape[0] != 1 and left.shape[1] != 1) or right.shape[1] != 1:
        raise ValueError(
            "incompatible matrix shapes " + str(left.shape) + " and " + str(right.shape)
        )


def inner_cupydense(left, right, scalar_is_ket=False):

    _check_shape_inner(left, right)

    if left.shape[0] == left.shape[1] == right.shape[1] == 1:
        if not cp.all(left._cp == 0) and not cp.all(right._cp == 0):
            return (
                (cp.conj(left._cp[0]) * right._cp[0]).item()
                if scalar_is_ket
                else (left._cp[0] * right._cp[0]).item()
            )
        return 0.0j

    if left.shape[0] == 1:
        # TODO check if this runs faster using qutip cublas methods
        # which allow to pass a flag to take conj
        return cp.dot(left._cp, right._cp).item()

    # TODO:remove the final .item when
    # possible as it forces CPU GPU  transfers.

    return cp.vdot(left._cp, right._cp).item()


def _check_shape_inner_op(left, op, right):
    left_shape = left.shape[0] == 1 or left.shape[1] == 1
    left_op = (left.shape[0] == 1 and left.shape[1] == op.shape[0]) or (
        left.shape[1] == 1 and left.shape[0] == op.shape[0]
    )
    op_right = op.shape[1] == right.shape[0]
    right_shape = right.shape[1] == 1
    if not (left_shape and left_op and op_right and right_shape):
        raise ValueError(
            "".join(
                [
                    "incompatible matrix shapes ",
                    str(left.shape),
                    ", ",
                    str(op.shape),
                    " and ",
                    str(right.shape),
                ]
            )
        )


def inner_op_cupydense(left, op, right, scalar_is_ket=False):

    _check_shape_inner_op(left, op, right)

    if (
        left.shape[0]
        == left.shape[1]
        == op.shape[0]
        == op.shape[1]
        == right.shape[1]
        == 1
    ):
        if cp.all(left._cp == 0) or cp.all(right._cp == 0) or cp.all(op._cp == 0):
            return 0
        lstate = cp.conj(left._cp[0]) if scalar_is_ket else left._cp[0]
        return (lstate * op._cp[0] * right._cp[0]).item()

    if left.shape[0] == 1:
        return cp.dot(left._cp, op._cp @ right._cp).item()
    return cp.vdot(left._cp, op._cp @ right._cp).item()


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
