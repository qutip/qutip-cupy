import cupy as cp

import pytest

from qutip_cupy import dense
from qutip_cupy import dense_functions as cdf
from qutip_cupy import CuPyDense

# from qutip.tests.core.data import conftest
import qutip.tests.core.data.test_mathematics as test_tools


# This are the global variables of the qutip test module
# by setting them in this way the value gets propagated to the abstract
# mixing which in turn propagates them to the mixing and finally sets
# the test cases when pytests are called
test_tools._ALL_CASES = {
    CuPyDense: lambda shape: [lambda: CuPyDense(cp.random.rand(*shape))]
}
test_tools._RANDOM = {
    CuPyDense: lambda shape: [lambda: CuPyDense(cp.random.rand(*shape))],
}
# @TODO:This one should be replaced by a complex number random generator in conf

# And now finally we get into the meat of the actual mathematical tests.


class TestAdd(test_tools.TestAdd):
    specialisations = [
        pytest.param(dense.add_cupydense, CuPyDense, CuPyDense, CuPyDense),
        pytest.param(dense.iadd_cupydense, CuPyDense, CuPyDense, CuPyDense),
    ]


class TestAdjoint(test_tools.TestAdjoint):

    specialisations = [
        pytest.param(dense.adjoint_cupydense, CuPyDense, CuPyDense),
    ]


class TestConj(test_tools.TestConj):

    specialisations = [
        pytest.param(dense.conj_cupydense, CuPyDense, CuPyDense),
    ]


class TestMatmul(test_tools.TestMatmul):

    specialisations = [
        pytest.param(dense.matmul_cupydense, CuPyDense, CuPyDense, CuPyDense),
    ]


class TestMul(test_tools.TestMul):

    specialisations = [
        pytest.param(dense.mul_cupydense, CuPyDense, CuPyDense),
    ]


class TestNeg(test_tools.TestNeg):

    specialisations = [
        pytest.param(dense.neg_cupydense, CuPyDense, CuPyDense),
    ]


class TestSub(test_tools.TestSub):
    specialisations = [
        pytest.param(dense.sub_cupydense, CuPyDense, CuPyDense, CuPyDense),
    ]


class TestTrace(test_tools.TestTrace):

    specialisations = [
        pytest.param(cdf.trace_cupydense, CuPyDense, complex),
    ]


class TestTranspose(test_tools.TestTranspose):

    specialisations = [
        pytest.param(dense.transpose_cupydense, CuPyDense, CuPyDense),
    ]
