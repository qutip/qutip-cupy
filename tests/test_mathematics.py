import cupy as cp

import pytest

from qutip_cupy import dense
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


# class TestInner(BinaryOpMixin):
#     # The inner product is a bit more specialist, since it has to handle inputs
#     # in a 1D space specially.  In order to keep things simple, we just
#     # generate those test cases completely separately from the standard
#     # `mathematically_correct`.

#     def op_numpy(self, left, right, scalar_is_ket=False):
#         if left.shape[1] == 1:
#             if left.shape[0] != 1 or scalar_is_ket:
#                 left = np.conj(left.T)
#         return (left @ right)[0, 0]

#     # These shapes are a little more non-standard.
#     _dim = 100
#     _scalar = pytest.param((1, 1), id="scalar")
#     _bra = pytest.param((1, _dim), id="bra")
#     _ket = pytest.param((_dim, 1), id="ket")
#     _op = pytest.param((_dim, _dim), id="square")
#     shapes = [
#         (_bra, _ket),
#         (_ket, _ket),
#     ]
#     bad_shapes = [
#         (_bra, _bra),
#         (_ket, _bra),
#         (_op, _ket),
#         (_op, _bra),
#         (_bra, _op),
#         (_ket, _op),
#     ]

#     specialisations = [
#         pytest.param(data.inner_csr, CSR, CSR, complex),
#     ]

#     def generate_scalar_is_ket(self, metafunc):
#         # For 1D subspaces, the special cases don't really matter since there's
#         # only really one type of matrix available, so this is parametrised
#         # with only case for each input argument.
#         parameters = (
#             ["op"]
#             + [x for x in metafunc.fixturenames if x.startswith("data_")]
#             + ["out_type"]
#         )
#         cases = []
#         for p_op in self.specialisations:
#             op, *types, out_type = p_op.values
#             args = (op, types, [(self._scalar, self._scalar)], out_type)
#             cases.extend(cases_type_shape_product(_RANDOM, *args))
#         metafunc.parametrize(parameters, cases)
#         metafunc.parametrize("scalar_is_ket", [True, False], ids=["ket", "bra"])

#     def test_scalar_is_ket(self, op, data_l, data_r, out_type, scalar_is_ket):
#         left, right = data_l(), data_r()
#         expected = self.op_numpy(left.to_array(), right.to_array(), scalar_is_ket)
#         test = op(left, right, scalar_is_ket)
#         assert isinstance(test, out_type)
#         if issubclass(out_type, Data):
#             assert test.shape == expected.shape
#             np.testing.assert_allclose(test.to_array(), expected, atol=self.tol)
#         else:
#             assert abs(test - expected) < self.tol


# class TestInnerOp(TernaryOpMixin):
#     # This is very very similar to TestInner.
#     def op_numpy(self, left, mid, right, scalar_is_ket=False):
#         if left.shape[1] == 1:
#             if left.shape[0] != 1 or scalar_is_ket:
#                 left = np.conj(left.T)
#         return (left @ mid @ right)[0, 0]

#     _dim = 100
#     _scalar = pytest.param((1, 1), id="scalar")
#     _bra = pytest.param((1, _dim), id="bra")
#     _ket = pytest.param((_dim, 1), id="ket")
#     _op = pytest.param((_dim, _dim), id="square")
#     shapes = [
#         (_bra, _op, _ket),
#         (_ket, _op, _ket),
#     ]
#     bad_shapes = [
#         (_bra, _op, _bra),
#         (_ket, _op, _bra),
#         (_op, _op, _ket),
#         (_op, _op, _bra),
#         (_bra, _op, _op),
#         (_ket, _op, _op),
#         (_bra, _bra, _ket),
#         (_ket, _bra, _ket),
#         (_bra, _ket, _ket),
#         (_ket, _ket, _ket),
#     ]

#     specialisations = [
#         pytest.param(data.inner_op_csr, CSR, CSR, CSR, complex),
#     ]

#     def generate_scalar_is_ket(self, metafunc):
#         parameters = (
#             ["op"]
#             + [x for x in metafunc.fixturenames if x.startswith("data_")]
#             + ["out_type"]
#         )
#         cases = []
#         for p_op in self.specialisations:
#             op, *types, out_type = p_op.values
#             args = (op, types, [(self._scalar,) * 3], out_type)
#             cases.extend(cases_type_shape_product(_RANDOM, *args))
#         metafunc.parametrize(parameters, cases)
#         metafunc.parametrize("scalar_is_ket", [True, False], ids=["ket", "bra"])

#     def test_scalar_is_ket(self, op, data_l, data_m, data_r, out_type, scalar_is_ket):
#         left, mid, right = data_l(), data_m(), data_r()
#         expected = self.op_numpy(
#             left.to_array(), mid.to_array(), right.to_array(), scalar_is_ket
#         )
#         test = op(left, mid, right, scalar_is_ket)
#         assert isinstance(test, out_type)
#         if issubclass(out_type, Data):
#             assert test.shape == expected.shape
#             np.testing.assert_allclose(test.to_array(), expected, atol=self.tol)
#         else:
#             assert abs(test - expected) < self.tol


# class TestKron(BinaryOpMixin):
#     def op_numpy(self, left, right):
#         return np.kron(left, right)

#     # Keep the dimension low because kron can get very expensive.
#     shapes = shapes_binary_unrestricted(dim=5)
#     bad_shapes = shapes_binary_bad_unrestricted(dim=5)
#     specialisations = [
#         pytest.param(data.kron_csr, CSR, CSR, CSR),
#     ]


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


# class TestProject(test_tools.Test):

#     specialisations = [
#         pytest.param(data.project_csr, CSR, CSR),
#     ]


class TestSub(test_tools.TestSub):
    specialisations = [
        pytest.param(dense.sub_cupydense, CuPyDense, CuPyDense, CuPyDense),
    ]


# class TestTrace(UnaryOpMixin):
#     def op_numpy(self, matrix):
#         return np.sum(np.diag(matrix))

#     shapes = [
#         (pytest.param((1, 1), id="1"),),
#         (pytest.param((100, 100), id="100"),),
#     ]
#     bad_shapes = [(x,) for x in shapes_unary() if x.values[0][0] != x.values[0][1]]
#     specialisations = [
#         pytest.param(data.trace_csr, CSR, complex),
#         pytest.param(data.trace_dense, Dense, complex),
#     ]

#     # Trace actually does have bad shape, so we put that in too.
#     def test_incorrect_shape_raises(self, op, data_m):
#         """
#         Test that the operation produces a suitable error if the shape is not a
#         square matrix.
#         """
#         with pytest.raises(ValueError):
#             op(data_m())


class TestTranspose(test_tools.TestTranspose):

    specialisations = [
        pytest.param(dense.transpose_cupydense, CuPyDense, CuPyDense),
    ]


# class TestProject(UnaryOpMixin):
#     def op_numpy(self, matrix):
#         if matrix.shape[0] == 1:
#             return np.outer(np.conj(matrix), matrix)
#         else:
#             return np.outer(matrix, np.conj(matrix))

#     shapes = [
#         (pytest.param((1, 1), id="scalar"),),
#         (pytest.param((1, 100), id="bra"),),
#         (pytest.param((100, 1), id="ket"),),
#     ]
#     bad_shapes = [
#         (pytest.param((10, 10), id="square"),),
#         (pytest.param((2, 10), id="nonsquare"),),
#     ]

#     specialisations = [
#         pytest.param(data.project_csr, CSR, CSR),
#         pytest.param(data.project_dense, Dense, Dense),
#     ]


# def _inv_dense(matrix):
#     # Add a diagonal so `matrix` is not singular
#     return data.inv_dense(
#         data.add(
#             matrix,
#             data.diag([1.1] * matrix.shape[0], shape=matrix.shape, dtype="dense"),
#         )
#     )


# def _inv_csr(matrix):
#     # Add a diagonal so `matrix` is not singular
#     return data.inv_csr(
#         data.add(
#             matrix, data.diag([1.1] * matrix.shape[0],
# shape=matrix.shape, dtype="csr")
#         )
#     )


# class TestInv(UnaryOpMixin):
#     def op_numpy(self, matrix):
#         return np.linalg.inv(matrix + np.eye(matrix.shape[0]) * 1.1)

#     shapes = [
#         (pytest.param((1, 1), id="scalar"),),
#         (pytest.param((10, 10), id="square"),),
#     ]
#     bad_shapes = [
#         (pytest.param((2, 10), id="nonsquare"),),
#         (pytest.param((1, 100), id="bra"),),
#         (pytest.param((100, 1), id="ket"),),
#     ]

#     specialisations = [
#         pytest.param(_inv_csr, CSR, CSR),
#         pytest.param(_inv_dense, Dense, Dense),
#     ]


# class TestSplitColumns(UnaryOpMixin):
#     # UnaryOpMixin
#     def op_numpy(self, matrix):
#         return [matrix[:, i].reshape((-1, 1)) for i in range(matrix.shape[1])]

#     shapes = [
#         (pytest.param((1, 1), id="scalar"),),
#         (pytest.param((10, 10), id="square"),),
#         (pytest.param((2, 10), id="nonsquare"),),
#         (pytest.param((1, 100), id="bra"),),
#         (pytest.param((100, 1), id="ket"),),
#     ]

#     specialisations = [
#         pytest.param(data.split_columns_csr, CSR, list),
#         pytest.param(data.split_columns_dense, Dense, list),
#     ]
