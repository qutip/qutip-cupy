"""
This module contains the ``CuPyDense`` class and associated function
conversion and specializations for registration with QuTiP's data layer.
"""

import numbers

import cupy as cp
from qutip.core import data


class CuPyDense(data.Data):
    """
    This class provides a dense matrix backend for QuTiP.
    Matrices are stored internally in a CuPy array on a GPU.
    If you have many GPUs you can set GPU ``i``
    by calling ``cp.cuda.Device(i).use()`` before construction.

    Parameters
    ----------
    data: array-like
        Data to be stored.
    shape: (int, int)
        Defaults to ``None``. If ``None`` will infer the shape from ``data``,
        else it will set the shape for the internal CuPy array.
    copy: bool
        Defaults to ``True``. Whether to make a copy of
        the elements in ``data`` or not.
    dtype:
        Data type specifier. Either ``cp.complex128`` or ``cp.complex64``
    """

    def __init__(self, data, shape=None, copy=True, dtype=cp.complex128):
        self.dtype = dtype
        base = cp.array(data, dtype=self.dtype, order="K", copy=copy)
        if shape is None:
            shape = base.shape
            # Promote to a ket by default if passed 1D data.
            if len(shape) == 1:
                shape = (shape[0], 1)
        if not (
            len(shape) == 2
            and isinstance(shape[0], numbers.Integral)
            and isinstance(shape[1], numbers.Integral)
            and shape[0] > 0
            and shape[1] > 0
        ):
            raise ValueError(
                f"shape must be a 2-tuple of positive ints, but is {shape!r}"
            )
        if shape and (shape[0] != base.shape[0] or shape[1] != base.shape[1]):
            if shape[0] * shape[1] != base.size:
                raise ValueError(
                    f"invalid shape {shape} for input data with size {base.shape}"
                )
            else:
                self._cp = base.reshape(shape)
        else:
            self._cp = base

        super().__init__((shape[0], shape[1]))

    @classmethod
    def _raw_cupy_constructor(cls, data):
        """
        A fast low-level constructor for wrapping an existing CuPy array in a
        CuPyDense object without copying it.

        The ``data`` argument must be a CuPy array with the correct shape.
        The CuPy array will not be copied and will be used as is.
        """
        out = cls.__new__(cls)
        super(cls, out).__init__(data.shape)
        out._cp = data
        out.dtype = data.dtype
        return out

    def copy(self):
        return self._raw_cupy_constructor(self._cp.copy())

    def to_array(self):
        return cp.asnumpy(self._cp)

    def conj(self):
        return CuPyDense._raw_cupy_constructor(self._cp.conj())

    def transpose(self):
        return CuPyDense._raw_cupy_constructor(self._cp.transpose())

    def adjoint(self):
        return CuPyDense._raw_cupy_constructor(self._cp.transpose().conj())

    def trace(self):
        return self._cp.trace()

    def __add__(left, right):
        if not isinstance(left, CuPyDense) or not isinstance(right, CuPyDense):
            return NotImplemented
        return CuPyDense._raw_cupy_constructor(cp.add(left._cp, right._cp))

    def __matmul__(left, right):
        if not isinstance(left, CuPyDense) or not isinstance(right, CuPyDense):
            return NotImplemented
        return CuPyDense._raw_cupy_constructor(left._cp @ right._cp)

    def __mul__(left, right):
        dense, number = (left, right) if isinstance(left, CuPyDense) else (right, left)
        if not isinstance(number, numbers.Number):
            return NotImplemented
        return CuPyDense._raw_cupy_constructor(dense._cp * complex(number))

    def __imul__(self, other):

        self._cp.__imul__(other)
        return self

    def __truediv__(left, right):
        dense, number = (left, right) if isinstance(left, CuPyDense) else (right, left)
        if not isinstance(number, numbers.Number):
            return NotImplemented
        return CuPyDense._raw_cupy_constructor(dense._cp.__truediv__(number))

    def __itruediv__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        self._cp.__itruediv__(other)
        return self

    def __neg__(self):

        return CuPyDense._raw_cupy_constructor(self._cp.__neg__())

    def __sub__(left, right):
        if not isinstance(left, CuPyDense) or not isinstance(right, CuPyDense):
            return NotImplemented
        return CuPyDense._raw_cupy_constructor(left._cp - right._cp)

    # def __dealloc__(self):
    #     if self._deallocate and self.data != NULL:
    #         PyDataMem_FREE(self.data)


# @TOCHECK  here I am reducing the aguments of empty but I should probably be keeping
# all of them at least as dummies
def empty(rows, cols, fortran):
    """
    Return a new Dense type of the given shape, with the data allocated but
    uninitialised.
    """
    order = "F" if fortran else "C"
    cparr = cp.empty(shape=(rows, cols), dtype=cp.complex128, order=order)
    return CuPyDense._raw_cupy_constructor(cparr)


def empty_like(other, fortran):
    """
    Return a new Dense type of the same shape as the given array.
    """
    order = "F" if fortran else "C"
    cparr = cp.empty_like(other, dtype=cp.complex128, order=order)
    return CuPyDense._raw_cupy_constructor(cparr)


def zeros(rows, cols, fortran):
    """Return the zero matrix with the given shape."""
    order = "F" if fortran else "C"
    cparr = cp.zeros(shape=(rows, cols), dtype=cp.complex128, order=order)
    return CuPyDense._raw_cupy_constructor(cparr)


def identity(dimension, scale=1, fortran=True):
    """
    Return a square matrix of the specified dimension, with a constant along
    the diagonal.  By default this will be the identity matrix, but if `scale`
    is passed, then the result will be `scale` times the identity.
    """
    order = "F" if fortran else "C"
    if scale != 1:
        cparr = cp.eye(dimension, dtype=cp.complex128, order=order)
    else:
        cparr = scale * cp.eye(dimension, dtype=cp.complex128, order=order)

    return CuPyDense._raw_cupy_constructor(cparr)


def Dense_from_csr(CSR_matrix, fortran=False):
    pass


def _diagonal_length(
    offset, n_rows, n_cols,
):
    if offset > 0:
        return n_rows if offset <= n_cols - n_rows else n_cols - offset
    return n_cols if offset > n_cols - n_rows else n_rows + offset


def diags(diagonals, offsets=None, shape=None):
    """
    Construct a matrix from diagonals and their offsets.  Using this
    function in single-argument form produces a square matrix with the given
    values on the main diagonal.
    With lists of diagonals and offsets, the matrix will be the smallest
    possible square matrix if shape is not given, but in all cases the
    diagonals must fit exactly with no extra or missing elements. Duplicated
    diagonals will be summed together in the output.

    Parameters
    ----------
    diagonals : sequence of array_like of complex or array_like of complex
        The entries (including zeros) that should be placed on the diagonals in
        the output matrix.  Each entry must have enough entries in it to fill
        the relevant diagonal and no more.
    offsets : sequence of integer or integer, optional
        The indices of the diagonals.  `offsets[i]` is the location of the
        values `diagonals[i]`.  An offset of 0 is the main diagonal, positive
        values are above the main diagonal and negative ones are below the main
        diagonal.
    shape : tuple, optional
        The shape of the output as (``rows``, ``columns``).  The result does
        not need to be square, but the diagonals must be of the correct length
        to fit in exactly.
    """
    # This implementation follows the one of cupy.diagonal
    # alternatively we may define our own cuda kernel.

    # we make sure to export the variables to the GPU
    # we should actually benchmark if we should ascertain
    # that sending the variables to the device is better before than after
    diagonals = cp.asarray(diagonals)
    offsets = cp.asarray(offsets)
    try:
        diagonals = list(diagonals)
        if diagonals and cp.isscalar(diagonals[0]):
            # Catch the case where we're being called as (for example)
            #   diags([1, 2, 3], 0)
            # with a single diagonal and offset.
            diagonals = diagonals[cp.newaxis, :]
    except TypeError:
        raise TypeError("diagonals must be a list of arrays of complex") from None
    diagonals_length = diagonals.shape[0]
    if offsets is None:
        if diagonals_length == 0:
            offsets = []
        elif diagonals_length == 1:
            offsets = [0]
        else:
            raise TypeError(
                "offsets must be supplied" "if passing more than one diagonal"
            )
    offsets = cp.atleast_1d(offsets)
    if offsets.ndim > 1:
        raise ValueError("offsets must be a 1D array of integers")
    if diagonals_length != len(offsets):
        raise ValueError("number of diagonals does not match number of offsets")
    if diagonals_length == 0:
        if shape is None:
            raise ValueError(
                "cannot construct matrix" "with no diagonals without a shape"
            )
        else:
            n_rows, n_cols = shape
        return zeros(n_rows, n_cols)
    # I am keeping this section assuming that properly sorting will lead to
    # contiguous memory accesses and a decreased runtime
    order = cp.argsort(offsets)
    diagonals_ = []
    offsets_ = []
    prev, cur = None, None
    for i in order:
        cur = offsets[i]
        if cur == prev:
            diagonals_[-1] += cp.asarray(diagonals[i], dtype=cp.complex128)
        else:
            offsets_.append(cur)
            diagonals_.append(cp.asarray(diagonals[i], dtype=cp.complex128))
        prev = cur
    if shape is None:
        n_rows = n_cols = abs(offsets_[0]) + len(diagonals_[0])
    else:
        try:
            n_rows, n_cols = shape
        except (TypeError, ValueError):
            raise TypeError("shape must be a 2-tuple of positive integers")
        if n_rows < 0 or n_cols < 0:
            raise ValueError("shape must be a 2-tuple of positive integers")
    for i in range(len(diagonals_)):
        offset = offsets_[i]
        if len(diagonals_[i]) != _diagonal_length(offset, n_rows, n_cols):
            raise ValueError("given diagonals do not have the correct lengths")
    if n_rows == 0 and n_cols == 0:
        raise ValueError("can't produce a 0x0 matrix")

    out = zeros(n_rows, n_cols, fortran=True)

    for diag_idx in diagonals_length:
        out.diagonal(offsets_[diag_idx])[:] = diagonals_[diag_idx]

    return out


# @TOCHECK I added docstrings describing functions as they are.
# If we were to have a precision parameter on the conversion
# I am not really sure how the dispatcher would handle it.
# It looks like we may be needing 2 classes.
def dense_from_cupydense(cupydense):
    """
    Creates a QuTiP ``data.Dense`` array from the values in a CuPyDense array.
    The resulting array has complex128 precision.
    """
    dense_np = data.Dense(cupydense.to_array(), copy=False)
    return dense_np


def cupydense_from_dense(dense):
    """
    Creates a CuPyDense array from the values in a QuTiP ``data.Dense`` array
    with ``cp.complex128`` precision.
    """
    dense_cp = CuPyDense(dense.as_ndarray(), copy=False)
    return dense_cp


def adjoint_cupydense(cpd_array):  # noqa: E302
    return cpd_array.adjoint()


def conj_cupydense(cpd_array):  # noqa: E302
    return cpd_array.conj()


def transpose_cupydense(cpd_array):  # noqa: E302
    return cpd_array.transpose()


def trace_cupydense(cpd_array):  # noqa: E302
    return cpd_array.trace()


def imul_cupydense(cpd_array, value):  # noqa: E302
    """Multiply this CuPyDense `cpd_array` by a complex scalar `value`."""
    cpd_array.__imul__(value)
    return cpd_array


def mul_cupydense(cpd_array, value):  # noqa: E302
    """Multiply this Dense `cpd_array` by a complex scalar `value`."""
    return cpd_array * value


def neg_cupydense(cpd_array):  # noqa: E302
    """Unary negation of this Dense `cpd_array`.  Return a new object."""
    return cpd_array.__neg__()


def matmul_cupydense(left, right, scale=1, out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1.
    """
    if out:
        return scale * (left @ right) + out
    else:
        return scale * (left @ right)


def add_cupydense(left, right, scale=1):
    """
    Perform the operation ``out := left + scale*right``
    Parameters
    ----------
    left : CuPyDense
        Matrix to be added.
    right : CuPyDense
        Matrix to be added.  If `scale` is given, this matrix will be
        multiplied by `scale` before addition.
    scale : optional double complex (1)
        The scalar value to multiply `right` by before addition.

    Returns
    -------
    out : CUPyDense
    """
    return left + scale * right


def sub_cupydense(left, right):
    """
    Perform the operation ``out := left - right``

    Parameters
    ----------
    left : CuPyDense
        Matrix to be added.
    right : CuPyDense
    Returns
    -------
    out : CUPyDense
    """
    return left - right
