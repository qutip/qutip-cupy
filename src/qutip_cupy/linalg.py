from .dense import CuPyDense

import cupy as cp


def inv_cupydense(data):
    """Compute the inverse of a matrix"""
    if not isinstance(data, CuPyDense):
        raise TypeError("expected data in Dense format but got " + str(type(data)))
    if data.shape[0] != data.shape[1]:
        raise ValueError("Cannot compute the matrix inverse" " of a nonsquare matrix")
    return CuPyDense._raw_cupy_constructor(cp.linalg.inv(data._cp))
