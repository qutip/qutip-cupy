import cupy as cp
import numbers
from qutip.core import data


class CuPyDense(data.Data):
    def __init__(self, data, shape=None, copy=True):
        base = cp.array(data, dtype=cp.complex128, order='K', copy=copy)
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
            raise ValueError("shape must be a 2-tuple of positive ints, but is " + repr(shape))
        if shape and (shape[0] != base.shape[0] or shape[1] != base.shape[1]):
            if shape[0] * shape[1] != base.size:
                raise ValueError("".join([
                    "invalid shape ",
                    str(shape),
                    " for input data with size ",
                    str(base.size)
                ]))
            else:
                self._cp = base.reshape(shape)
        else:

            self._cp = base

        super().__init__((shape[0], shape[1]))

    def copy(self):
        return CuPyDense(self._cp.copy())

    def to_array(self):
        """ 
        Get a copy as a `numpy.ndarray`.
        This incurs memory-transfer from `host` to `device`
        and should be avoided when possible    
        """
        return cp.asnumpy(self._cp)

    def conj(self):
        return CuPyDense(self._cp.conj())

    def transpose(self):
        return CuPyDense(self._cp.transpose())

    def adjoint(self):
        return CuPyDense(self._cp.transpose().conj())

    def trace(self):
        return self._cp.trace()

    def __add__(left, right):
        if not isinstance(left, Dense) or not isinstance(right, Dense):
            return NotImplemented
        return CuPyDense(cp.add(left._cp, right._cp))

    def __matmul__(left, right):
        if not isinstance(left, Dense) or not isinstance(right, Dense):
            return NotImplemented
        return CuPyDense(cp.matmul(left._cp, right._cp))

    def __mul__(left, right):
        dense, number = (left, right) if isinstance(left, Dense) else (right, left)
        if not isinstance(number, numbers.Number):
            return NotImplemented
        return CuPyDense(dense._cp * complex(number))

    # def __imul__(self, other):
    #     if not isinstance(other, numbers.Number):
    #         return NotImplemented
    #     cdef int size = self.shape[0]*self.shape[1]
    #     cdef double complex mul = complex(other)
    #     blas.zscal(&size, &mul, self.data, &_ONE)
    #     return self

    def __truediv__(left, right):
        dense, number = (left, right) if isinstance(left, Dense) else (right, left)
        if not isinstance(number, numbers.Number):
            return NotImplemented
        # Technically `(1 / x) * y` doesn't necessarily equal `y / x` in
        # floating point, but multiplication is faster than division, and we
        # don't really care _that_ much anyway.
        return mul_dense(dense, 1 / complex(number))

    def __itruediv__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        cdef int size = self.shape[0]*self.shape[1]
        cdef double complex mul = 1 / complex(other)
        blas.zscal(&size, &mul, self.data, &_ONE)
        return self

    def __neg__(self):
        return neg_dense(self)

    def __sub__(left, right):
        if not isinstance(left, Dense) or not isinstance(right, Dense):
            return NotImplemented
        return sub_dense(left, right)

    def __dealloc__(self):
        if self._deallocate and self.data != NULL:
            PyDataMem_FREE(self.data)


#@TOCHECK  here I am reducing the aguments of empty but I should probably be keeping all of them at least as dummies

def empty(rows, cols):
    """
    Return a new Dense type of the given shape, with the data allocated but
    uninitialised.
    """
    # cdef Dense out = Dense.__new__(Dense)
    # out.shape = (rows, cols)
    # out.data = <double complex *> PyDataMem_NEW(rows * cols * sizeof(double complex))
    # out._deallocate = True
    # out.fortran = fortran
    return CuPyDense(cp.empty(shape=(rows, cols), dtype=cp.complex128))


def empty_like(other):
    return CuPyDense(cp.empty_like(other))


def zeros(rows, cols, fortran):
    """Return the zero matrix with the given shape."""
    # cdef Dense out = Dense.__new__(Dense)
    # out.shape = (rows, cols)
    # out.data =\
    #     <double complex *> PyDataMem_NEW_ZEROED(rows * cols, sizeof(double complex))
    # out.fortran = fortran
    # out._deallocate = True
    contiguity = 'F' if fortran else 'C'
    out = empty(rows, cols, contiguity)
    out._cp = cp.zeros(shape=(rows, cols), dtype=cp.complex128, order=contiguity)
    return out


def identity(dimension, scale=1, fortran=True):
    """
    Return a square matrix of the specified dimension, with a constant along
    the diagonal.  By default this will be the identity matrix, but if `scale`
    is passed, then the result will be `scale` times the identity.
    """
    out = cp.eye(dimension, dtype=cp.complex128) if scale != 1 else scale*cp.eye(dimension, dtype=cp.complex128)
    return CuPyDense(out)


def Dense from_csr(CSR matrix, bint fortran=False):
    cdef Dense out = Dense.__new__(Dense)
    out.shape = matrix.shape
    out.data = (
        <double complex *>
        PyDataMem_NEW_ZEROED(out.shape[0]*out.shape[1], sizeof(double complex))
    )
    out.fortran = fortran
    out._deallocate = True
    cdef size_t row, ptr_in, ptr_out, row_stride, col_stride
    row_stride = 1 if fortran else out.shape[1]
    col_stride = out.shape[0] if fortran else 1
    ptr_out = 0
    for row in range(out.shape[0]):
        for ptr_in in range(matrix.row_index[row], matrix.row_index[row + 1]):
            out.data[ptr_out + matrix.col_index[ptr_in]*col_stride] = matrix.data[ptr_in]
        ptr_out += row_stride
    return out


# def inline base.idxint _diagonal_length(
#     base.idxint offset, base.idxint n_rows, base.idxint n_cols,
# ) nogil:
#     if offset > 0:
#         return n_rows if offset <= n_cols - n_rows else n_cols - offset
#     return n_cols if offset > n_cols - n_rows else n_rows + offset

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
    cdef base.idxint n_rows, n_cols, offset
    try:
        diagonals = list(diagonals)
        if diagonals and np.isscalar(diagonals[0]):
            # Catch the case where we're being called as (for example)
            #   diags([1, 2, 3], 0)
            # with a single diagonal and offset.
            diagonals = [diagonals]
    except TypeError:
        raise TypeError("diagonals must be a list of arrays of complex") from None
    if offsets is None:
        if len(diagonals) == 0:
            offsets = []
        elif len(diagonals) == 1:
            offsets = [0]
        else:
            raise TypeError("offsets must be supplied if passing more than one diagonal")
    offsets = np.atleast_1d(offsets)
    if offsets.ndim > 1:
        raise ValueError("offsets must be a 1D array of integers")
    if len(diagonals) != len(offsets):
        raise ValueError("number of diagonals does not match number of offsets")
    if len(diagonals) == 0:
        if shape is None:
            raise ValueError("cannot construct matrix with no diagonals without a shape")
        else:
            n_rows, n_cols = shape
        return zeros(n_rows, n_cols)
    order = np.argsort(offsets)
    diagonals_ = []
    offsets_ = []
    prev, cur = None, None
    for i in order:
        cur = offsets[i]
        if cur == prev:
            diagonals_[-1] += np.asarray(diagonals[i], dtype=np.complex128)
        else:
            offsets_.append(cur)
            diagonals_.append(np.asarray(diagonals[i], dtype=np.complex128))
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

    cdef size_t diag_idx, idx, n_diagonals = len(diagonals_)

    for diag_idx in range(n_diagonals):
        offset = offsets_[diag_idx]
        if offset <= 0:
            for idx in range(_diagonal_length(offset, n_rows, n_cols)):
                out.data[idx*(n_rows+1) - offset] = diagonals_[diag_idx][idx]
        else:
            for idx in range(_diagonal_length(offset, n_rows, n_cols)):
                out.data[idx*(n_rows+1) + offset*n_rows] = diagonals_[diag_idx][idx]
    return out



def dense_from_cupydense(cupydense):

    dense_np = data.Dense(cupydense.to_array(), copy=False)
    return dense_np


def cupydense_from_dense(dense):

    dense_cp = CuPyDense(dense.as_ndarray(), copy=False)
    return dense_cp


def cpd_adjoint(cpd_array):
    return cpd_array.adjoint()


def cpd_conj(cpd_array):
    return cpd_array.conj()


def cpd_transpose(cpd_array):
    return cpd_array.transpose()


def cpd_trace(cpd_array):
    return cpd_array.trace()
