import cupy as cp
import numpy as np
import pytest

from qutip_cupy import dense
from qutip_cupy import dense_functions as cdf
from qutip_cupy import linalg
from qutip_cupy import CuPyDense

import qutip.tests.core.data.test_mathematics as test_tools
from qutip.core.data import Data


def random_cupydense(shape):
    """Generate a random `CuPyDense` matrix with the given shape."""
    out = (cp.random.rand(*shape) + 1j * cp.random.rand(*shape)).astype(cp.complex128)
    out = CuPyDense._raw_cupy_constructor(out)
    return out


# This are the global variables of the qutip test module
# by setting them in this way the value gets propagated to the abstract
# mixing which in turn propagates them to the mixing and finally sets
# the test cases when pytests are called
test_tools._ALL_CASES = {CuPyDense: lambda shape: [lambda: random_cupydense(shape)]}
test_tools._RANDOM = {
    CuPyDense: lambda shape: [lambda: random_cupydense(shape)],
}
# @TODO: add a simple precision complex random generator.


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


class TestHerm:
    tol = 1e-12

    @pytest.mark.parametrize("size", (10, 20, 100, 1000))
    def test_random_equal_structure(self, size):

        # Complex Hermitian matrices
        base = (
            np.random.uniform(size=(size, size))
            + np.random.uniform(size=(size, size)) * 1j
        ).astype(np.complex128)
        base += base.T.conj()
        base = CuPyDense(base)
        assert cdf.isherm_cupydense(base, self.tol)

        # Complex skew-Hermitian matrices
        base = (
            np.random.uniform(size=(size, size))
            + np.random.uniform(size=(size, size)) * 1.0j
        ).astype(np.complex128)
        base += base.T
        base = CuPyDense(base)
        assert not cdf.isherm_cupydense(base, self.tol)

    @pytest.mark.parametrize("cols", (2, 4))
    @pytest.mark.parametrize("rows", (1, 5))
    def test_nonsquare_shapes(self, rows, cols):
        base = (
            np.random.uniform(size=(rows, cols))
            + np.random.uniform(size=(rows, cols)) * 1.0j
        ).astype(np.complex128)
        base = CuPyDense(base)
        assert not cdf.isherm_cupydense(base, self.tol)
        assert not cdf.isherm_cupydense(base.transpose(), self.tol)

    def test_diagonal_elements(self):
        n = 10
        base = CuPyDense(np.diag(np.random.rand(n)))
        assert cdf.isherm_cupydense(base, tol=self.tol)
        assert not cdf.isherm_cupydense(base * 1j, tol=self.tol)


class TestSplitColumns(test_tools.UnaryOpMixin):
    def op_numpy(self, matrix, copy=True):
        return [
            np.array(matrix[:, k], copy=copy).reshape(matrix.shape[0], 1)
            for k in range(matrix.shape[1])
        ]

    specialisations = [
        pytest.param(cdf.split_columns_cupydense, CuPyDense, list),
    ]


class TestInner(test_tools.TestInner):

    specialisations = [
        pytest.param(cdf.inner_cupydense, CuPyDense, CuPyDense, complex),
    ]


class TestInnerOp(test_tools.TestInnerOp):

    specialisations = [
        pytest.param(cdf.inner_op_cupydense, CuPyDense, CuPyDense, CuPyDense, complex),
    ]


class TestKron(test_tools.TestKron):

    specialisations = [
        pytest.param(cdf.kron_cupydense, CuPyDense, CuPyDense, CuPyDense),
    ]


class TestFrobeniusNorm(test_tools.UnaryOpMixin):
    # TODO add this tests to QuTiP and then inherit
    def op_numpy(self, matrix):
        return np.linalg.norm(matrix)

    shapes = [
        (pytest.param((1, 1), id="1"),),
        (pytest.param((100, 100), id="100"),),
        (pytest.param((100, 1), id="100_ket"),),
        (pytest.param((1, 100), id="100_bra"),),
        (pytest.param((23, 30), id="23_30"),),
    ]

    specialisations = [
        pytest.param(cdf.frobenius_cupydense, CuPyDense, float),
    ]


class TestL2Norm(test_tools.UnaryOpMixin):
    # TODO add this tests to QuTiP and then inherit
    def op_numpy(self, matrix):
        return np.linalg.norm(matrix)

    shapes = [
        (pytest.param((1, 1), id="1"),),
        (pytest.param((100, 1), id="20_ket"),),
        (pytest.param((1, 100), id="10_bra"),),
    ]

    bad_shapes = [
        (pytest.param((100, 100), id="100"),),
        (pytest.param((23, 30), id="23_30"),),
        (pytest.param((15, 10), id="15_10"),),
    ]

    specialisations = [
        pytest.param(cdf.l2_cupydense, CuPyDense, float),
    ]

    # l2 norm actually does have bad shape, so we put that in too.
    def test_incorrect_shape_raises(self, op, data_m):
        """
        Test that the operation produces a suitable error if the shape is not a
        bra or ket.
        """
        with pytest.raises(ValueError):
            op(data_m())


class TestMaxNorm(test_tools.UnaryOpMixin):
    # TODO add this tests to QuTiP and then inherit
    def op_numpy(self, matrix):
        return np.max(np.abs(matrix))

    shapes = [
        (pytest.param((1, 1), id="1"),),
        (pytest.param((100, 100), id="100"),),
        (pytest.param((100, 1), id="100_ket"),),
        (pytest.param((1, 100), id="100_bra"),),
        (pytest.param((23, 30), id="23_30"),),
    ]

    specialisations = [
        pytest.param(cdf.max_cupydense, CuPyDense, float),
    ]


class TestL1Norm(test_tools.UnaryOpMixin):
    # TODO add this tests to QuTiP and then inherit
    def op_numpy(self, matrix):
        return np.linalg.norm(matrix, ord=1)

    shapes = [
        (pytest.param((1, 1), id="1"),),
        (pytest.param((100, 100), id="100"),),
        (pytest.param((100, 1), id="100_ket"),),
        (pytest.param((1, 100), id="100_bra"),),
        (pytest.param((23, 30), id="23_30"),),
    ]

    specialisations = [
        pytest.param(cdf.one_cupydense, CuPyDense, float),
    ]


class TestPow(test_tools._GenericOpMixin):

    # This should be part of QuTiP and only inherited here
    # TODO: Add it to QuTiP and inherit here

    def op_numpy(self, matrix, scalar):

        return np.linalg.matrix_power(matrix, scalar)

    shapes = [
        (pytest.param((1, 1),),),
        (pytest.param((5, 5),),),
        (pytest.param((10, 10),),),
    ]

    bad_shapes = [
        (x,) for x in test_tools.shapes_unary() if x.values[0][0] != x.values[0][1]
    ]

    specialisations = [
        pytest.param(cdf.pow_cupydense, CuPyDense, CuPyDense),
    ]

    @pytest.mark.parametrize(
        "scalar",
        [pytest.param(1, id="1"), pytest.param(2, id="2"), pytest.param(5, id="5")],
    )
    def test_mathematically_correct(self, op, data_m, scalar, out_type):
        matrix = data_m()
        expected = self.op_numpy(matrix.to_array(), scalar)
        test = op(matrix, scalar)
        assert isinstance(test, out_type)
        if issubclass(out_type, Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected, atol=self.tol)
        else:
            assert abs(test - expected) < self.tol

    @pytest.mark.parametrize(
        "scalar",
        [pytest.param(1, id="1"), pytest.param(2, id="2"), pytest.param(5, id="5")],
    )
    def test_incorrect_shape_raises(self, op, data_m, scalar):
        """
        Test that the operation produces a suitable error if the matrices are not
        square.
        """
        with pytest.raises(ValueError):
            op(data_m(), scalar)


class TestProject(test_tools.TestProject):

    specialisations = [
        pytest.param(cdf.project_cupydense, CuPyDense, CuPyDense),
    ]


def _inv_cpd(matrix):
    # Add a diagonal so `matrix` is not singular
    return linalg.inv_cupydense(
        matrix + dense.diags([1.1] * matrix.shape[0], [0], shape=matrix.shape)
    )


class TestInv(test_tools.TestInv):

    specialisations = [
        pytest.param(_inv_cpd, CuPyDense, CuPyDense),
    ]
