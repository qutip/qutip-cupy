from qutip_cupy import CuPyDense
from qutip_cupy import dense
import numpy as np
import pytest


@pytest.fixture(scope="function", params=((1, 2), (5, 10), (7, 3), (2, 5)))
def shape(request):
    return request.param


class TestCuPyDenseDispatch:
    """ Tests if the methods and conversions have been
        succesfully registered to QuTiP's Data Layer."""

    def test_conversion_cycle(self, shape):

        from qutip.core import data

        qutip_dense = data.Dense(np.random.uniform(size=shape))

        tr1 = data.to(CuPyDense, qutip_dense)
        tr2 = data.to(data.Dense, tr1)

        np.testing.assert_array_equal(qutip_dense.to_array(), tr2.to_array())


class TestCuPyDense:
    """ Tests of the methods and constructors of the CuPyDense class. """

    def test_shape(self, shape):
        cupy_dense = CuPyDense(np.random.uniform(size=shape))
        assert cupy_dense.shape == shape

    def test_transpose(self, shape):
        cupy_dense = CuPyDense(np.random.uniform(size=shape)).transpose()
        np.testing.assert_array_equal(cupy_dense.shape, (shape[1], shape[0]))

    def test_adjoint(self, shape):
        data = np.random.uniform(size=shape) + 1.0j * np.random.uniform(size=shape)
        cpd_adj = CuPyDense(data).adjoint()
        np.testing.assert_array_equal(cpd_adj.to_array(), data.transpose().conj())

    @pytest.mark.parametrize(
        ["matrix", "trace"],
        [
            pytest.param([[0, 1], [1, 0]], 0),
            pytest.param([[2.0j, 1], [1, 1]], 1 + 2.0j),
        ],
    )
    def test_trace(self, matrix, trace):
        cupy_array = CuPyDense(matrix)
        assert cupy_array.trace() == trace


def test_true_div(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.0j * np.random.uniform(size=shape)

    cup_arr = CuPyDense(array)
    cpdense_tr = cup_arr / 2.0
    qtpdense_tr = data.Dense(array) / 2.0

    np.testing.assert_array_equal(cpdense_tr.to_array(), qtpdense_tr.to_array())


def test_itrue_div(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.0j * np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__itruediv__(2.0)
    qtpdense_tr = data.Dense(array).__itruediv__(2.0)

    np.testing.assert_array_equal(cpdense_tr.to_array(), qtpdense_tr.to_array())


def test_mul(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.0j * np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__mul__(2.0 + 1.0j)
    qtpdense_tr = data.Dense(array).__mul__(2.0 + 1.0j)

    np.testing.assert_array_equal(cpdense_tr.to_array(), qtpdense_tr.to_array())


def test_matmul(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.0j * np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__mul__(2.0 + 1.0j)
    qtpdense_tr = data.Dense(array).__mul__(2.0 + 1.0j)

    np.testing.assert_array_equal(cpdense_tr.to_array(), qtpdense_tr.to_array())


class TestFactoryMethods:
    def test_empty(self, shape):
        base = dense.empty(shape[0], shape[1])
        nd = base.to_array()
        assert isinstance(base, CuPyDense)
        assert base.shape == shape
        assert nd.shape == shape

    def test_zeros(self, shape):
        base = dense.zeros(shape[0], shape[1])
        nd = base.to_array()
        assert isinstance(base, CuPyDense)
        assert base.shape == shape
        assert nd.shape == shape
        assert np.count_nonzero(nd) == 0

    @pytest.mark.parametrize("dimension", [1, 5, 100])
    @pytest.mark.parametrize(
        "scale",
        [None, 2, -0.1, 1.5, 1.5 + 1j],
        ids=["none", "int", "negative", "float", "complex"],
    )
    def test_identity(self, dimension, scale):
        # scale=None is testing that the default value returns the identity.
        base = (
            dense.identity(dimension)
            if scale is None
            else dense.identity(dimension, scale)
        )
        nd = base.to_array()
        numpy_test = np.eye(dimension, dtype=np.complex128)
        if scale is not None:
            numpy_test *= scale
        assert isinstance(base, CuPyDense)
        assert base.shape == (dimension, dimension)
        assert np.count_nonzero(nd - numpy_test) == 0

    @pytest.mark.parametrize(
        ["diagonals", "offsets", "shape"],
        [
            pytest.param([2j, 3, 5, 9], None, None, id="main diagonal"),
            pytest.param([1], None, None, id="1x1"),
            pytest.param([[0.2j, 0.3]], None, None, id="main diagonal list"),
            pytest.param([0.2j, 0.3], 2, None, id="superdiagonal"),
            pytest.param([0.2j, 0.3], -2, None, id="subdiagonal"),
            pytest.param(
                [[0.2, 0.3, 0.4], [0.1, 0.9]], [-2, 3], None, id="two diagonals"
            ),
            pytest.param([1, 2, 3], 0, (3, 5), id="main wide"),
            pytest.param([1, 2, 3], 0, (5, 3), id="main tall"),
            pytest.param([[1, 2, 3], [4, 5]], [-1, -2], (4, 8), id="two wide sub"),
            pytest.param(
                [[1, 2, 3, 4], [4, 5, 4j, 1j]], [1, 2], (4, 8), id="two wide super"
            ),
            pytest.param([[1, 2, 3], [4, 5]], [1, 2], (8, 4), id="two tall super"),
            pytest.param(
                [[1, 2, 3, 4], [4, 5, 4j, 1j]], [-1, -2], (8, 4), id="two tall sub"
            ),
            pytest.param(
                [[1, 2, 3], [4, 5, 6], [1, 2]], [1, -1, -2], (4, 4), id="out of order"
            ),
            pytest.param(
                [[1, 2, 3], [4, 5, 6], [1, 2]], [1, 1, -2], (4, 4), id="sum duplicates"
            ),
        ],
    )
    def test_diags(self, diagonals, offsets, shape):
        base = dense.diags(diagonals, offsets, shape)
        # Build numpy version test.
        if not isinstance(diagonals[0], list):
            diagonals = [diagonals]
        offsets = np.atleast_1d(offsets if offsets is not None else [0])
        if shape is None:
            size = len(diagonals[0]) + abs(offsets[0])
            shape = (size, size)
        test = np.zeros(shape, dtype=np.complex128)
        for diagonal, offset in zip(diagonals, offsets):
            test[np.where(np.eye(*shape, k=offset) == 1)] += diagonal
        assert isinstance(base, CuPyDense)
        assert base.shape == shape
        np.testing.assert_allclose(base.to_array(), test, rtol=1e-10)

    # @pytest.mark.parametrize(
    #     ["shape", "position", "value"],
    #     [
    #         pytest.param((1, 1), (0, 0), None, id="minimal"),
    #         pytest.param((10, 10), (5, 5), 1.0j, id="on diagonal"),
    #         pytest.param((10, 10), (1, 5), 1.0, id="upper"),
    #         pytest.param((10, 10), (5, 1), 2.0, id="lower"),
    #         pytest.param((10, 1), (5, 0), None, id="column"),
    #         pytest.param((1, 10), (0, 5), -5.0j, id="row"),
    #         pytest.param((10, 2), (5, 1), 1 + 2j, id="tall"),
    #         pytest.param((2, 10), (1, 5), 10, id="wide"),
    #     ],
    # )
    # def test_one_element(self, shape, position, value):
    #     test = np.zeros(shape, dtype=np.complex128)
    #     if value is None:
    #         base = data.one_element_dense(shape, position)
    #         test[position] = 1.0 + 0.0j
    #     else:
    #         base = data.one_element_dense(shape, position, value)
    #         test[position] = value
    #     assert isinstance(base, CuPyDense)
    #     assert base.shape == shape
    #     assert np.allclose(base.to_array(), test, atol=1e-10)

    # @pytest.mark.parametrize(
    #     ["shape", "position", "value"],
    #     [
    #         pytest.param((0, 0), (0, 0), None, id="zero shape"),
    #         pytest.param((10, -2), (5, 0), 1.0j, id="neg shape"),
    #         pytest.param((10, 10), (10, 5), 1.0, id="outside"),
    #         pytest.param((10, 10), (5, -1), 2.0, id="outside neg"),
    #     ],
    # )
    # def test_one_element_error(self, shape, position, value):
    #     with pytest.raises(ValueError) as exc:
    #         base = data.one_element_dense(shape, position, value)
    #     assert str(exc.value).startswith("Position of the elements" " out of bound: ")

