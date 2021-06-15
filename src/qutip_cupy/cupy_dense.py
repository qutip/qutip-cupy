import cupy as cp
import numbers

class NewDataType:
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
        if shape[0] * shape[1] != base.size:
            raise ValueError("".join([
                "invalid shape ",
                str(shape),
                " for input data with size ",
                str(base.size)
            ]))
        self._cp = base      # self._deallocate = False
       # self.data = <double complex *> cnp.PyArray_GETPTR2(self._np, 0, 0)
       # self.fortran = cnp.PyArray_IS_F_CONTIGUOUS(self._np)
        self.shape = (shape[0], shape[1])

