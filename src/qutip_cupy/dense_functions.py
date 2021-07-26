# Contains functions for dense_cupy, this are the same functions that
# are defined ouside the dense.pyx file

import cupy as cp

from . import CuPyDense


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


def kron_cupydense(cp_arr, idxint, inplace=False):

    # Any improvement on matmul could be applied here since
    # cp.kron depends on cp.cupy._core.tensordot_core
    pass


def trace_cupydense(cp_arr):

    return cp.trace(cp_arr._cp)


def ptrace_cupydense(cp_arr, dims, sel):

    #  _check_shape(matrix)
    # dims, sel = _prepare_inputs(dims, sel)
    # if len(sel) == len(dims):
    #     return matrix.copy()
    # nd = dims.shape[0]
    # dkeep = [dims[x] for x in sel]
    # qtrace = list(set(np.arange(nd)) - set(sel))
    # dtrace = [dims[x] for x in qtrace]
    # dims = list(dims)
    # sel = list(sel)
    # rhomat = np.trace(matrix.as_ndarray()
    #                   .reshape(dims + dims)
    #                   .transpose(qtrace + [nd + q for q in qtrace] +
    #                              sel + [nd + q for q in sel])
    #                   .reshape([np.prod(dtrace),
    #                             np.prod(dtrace),
    #                             np.prod(dkeep),
    #                             np.prod(dkeep)]))
    #
    # return CuPyDense._raw_cupy_constructor(rhomat)
    pass

    # return cp.trace(cp_arr)


# reshape
# stack
# kron

# dot

# eigval
# pow
# expm
