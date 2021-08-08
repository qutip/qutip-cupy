# import cupy as cp
# import pytest

# from qutip_cupy import CuPyDense
# from qutip_cupy.expectation import expect_cupydense

# import qutip.tests.core.data.test_expect as test_tools


# def random_cupydense(shape):
#     """Generate a random `CuPyDense` matrix with the given shape."""
#     out = (cp.random.rand(*shape) + 1j * cp.random.rand(*shape)).astype(cp.complex128)
#     out = CuPyDense._raw_cupy_constructor(out)
#     return out


# class TestExpect(test_tools.TestExpect):

#     specialisations = [
#         pytest.param(expect_cupydense, CuPyDense, CuPyDense, complex),
#     ]
