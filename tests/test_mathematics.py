import numpy as np
import cupy as cp
import pytest

from qutip_cupy import dense


from qutip_cupy import CuPyDense

# from qutip.tests.core.data import conftest
import qutip.tests.core.data.test_mathematics as test_tools

# from qutip.tests.core.data.test_mathematics import (
#     shape_unary,
#     shapes_binary_identical,
#     shapes_binary_bad_identical,
#     shapes_binary_unrestricted,
#     shapes_binary_bad_unrestricted,
#     shapes_binary_matmul,
#     shapes_binary_bad_matmul,
#     UnaryOpMixin,
#     UnaryScalarOpMixin,
#     BinaryOpMixin,
#     TernaryOpMixin,
# )


# def cases_cupydense(shape):
#     """
#     Return a list of generators of the different special cases for Dense
#     matrices of a given shape.
#     """

#     def factory(fortran):
#         return lambda: conftest.random_dense(shape, fortran)

#     return [
#         pytest.param(factory(False), id="C"),
#         pytest.param(factory(True), id="Fortran"),
#     ]


# Factory methods for generating the cases, mapping type to the function.
# _ALL_CASES is for getting all the special cases to test, _RANDOM is for
# getting just a single case from each.
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


# class TestAdd(test_tools.BinaryOpMixin):
#     def op_numpy(self, left, right, scale):
#         return np.add(left, scale * right)

#     shapes = shapes_binary_identical()
#     bad_shapes = shapes_binary_bad_identical()
#     specialisations = [
#         pytest.param(data.add_dense, CuPyDense, CuPyDense, CuPyDense),
#     ]

#     # `add` has an additional scalar parameter, because the operation is
#     # actually more like `A + c*B`.  We just parametrise that scalar
#     # separately.
#     @pytest.mark.parametrize(
#         "scale", [None, 0.2, 0.5j], ids=["unscaled", "scale[real]", "scale[complex]"]
#     )
#     def test_mathematically_correct(self, op, data_l, data_r, out_type, scale):
#         """
#         Test that the binary operation is mathematically correct for all the
#         known type specialisations.
#         """
#         left, right = data_l(), data_r()
#         if scale is not None:
#             expected = self.op_numpy(left.to_array(), right.to_array(), scale)
#             test = op(left, right, scale)
#         else:
#             expected = self.op_numpy(left.to_array(), right.to_array(), 1)
#             test = op(left, right)
#         assert isinstance(test, out_type)
#         if issubclass(out_type, Data):
#             assert test.shape == expected.shape
#             np.testing.assert_allclose(test.to_array(), expected, atol=self.tol)
#         else:
#             assert abs(test - expected) < self.tol


# class TestAdjoint(test_tools.UnaryOpMixin):
#     def op_numpy(self, matrix):
#         return np.conj(matrix.T)

#     specialisations = [
#         pytest.param(data.adjoint_dense, CuPyDense, CuPyDense),
#     ]


# class TestConj(test_tools.UnaryOpMixin):
#     def op_numpy(self, matrix):
#         return np.conj(matrix)

#     specialisations = [
#         pytest.param(data.conj_dense, CuPyDense, CuPyDense),
#     ]


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


# class TestMatmul(BinaryOpMixin):
#     def op_numpy(self, left, right):
#         return np.matmul(left, right)

#     shapes = shapes_binary_matmul()
#     bad_shapes = shapes_binary_bad_matmul()
#     specialisations = [
#         pytest.param(data.matmul_csr, CSR, CSR, CSR),
#         pytest.param(data.matmul_csr_dense_dense, CSR, Dense, Dense),
#         pytest.param(data.matmul_dense, Dense, Dense, Dense),
#     ]


# class TestMul(UnaryScalarOpMixin):
#     def op_numpy(self, matrix, scalar):
#         return scalar * matrix

#     specialisations = [
#         pytest.param(data.mul_csr, CSR, CSR),
#         pytest.param(data.mul_dense, Dense, Dense),
#     ]


# class TestNeg(UnaryOpMixin):
#     def op_numpy(self, matrix):
#         return -matrix

#     specialisations = [
#         pytest.param(data.neg_csr, CSR, CSR),
#         pytest.param(data.neg_dense, Dense, Dense),
#     ]


# class TestProject(UnaryOpMixin):
#     def op_numpy(self, matrix):
#         if matrix.shape[1] == 1:
#             matrix = np.conj(matrix.T)
#         return np.conj(matrix.T) @ matrix

#     shapes = [
#         (pytest.param((1, 100), id="bra"),),
#         (pytest.param((100, 1), id="ket"),),
#         (pytest.param((1, 1), id="scalar"),),
#     ]
#     specialisations = [
#         pytest.param(data.project_csr, CSR, CSR),
#     ]


# class TestSub(BinaryOpMixin):
#     def op_numpy(self, left, right):
#         return left - right

#     shapes = shapes_binary_identical()
#     bad_shapes = shapes_binary_bad_identical()
#     specialisations = [
#         pytest.param(data.sub_csr, CSR, CSR, CSR),
#         pytest.param(data.sub_dense, Dense, Dense, Dense),
#     ]


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


# class TestTranspose(UnaryOpMixin):
#     def op_numpy(self, matrix):
#         return matrix.T

#     specialisations = [
#         pytest.param(data.transpose_csr, CSR, CSR),
#         pytest.param(data.transpose_dense, Dense, Dense),
#     ]


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
#             matrix, data.diag([1.1] * matrix.shape[0], shape=matrix.shape, dtype="csr")
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

