from qutip_cupy import CuPyDense
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

        assert (cupy_dense.shape == shape)

    def test_transpose(self, shape):

        cupy_dense = CuPyDense(np.random.uniform(size=shape)).transpose()
        
        np.testing.assert_array_equal(cupy_dense.shape, (shape[1],shape[0]))

    def test_adjoint(self, shape):

        from qutip.core import data

        array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

        cpdense_adj = CuPyDense(array).adjoint()
        qtpdense_adj = data.Dense(array).adjoint()

        np.testing.assert_array_equal(cpdense_adj.to_array(), qtpdense_adj.to_array())

    @pytest.mark.parametrize(["matrix", "trace"], [pytest.param([[0, 1],[1, 0]], 0),
                                                pytest.param([[2.j, 1],[1, 1]], 1+2.j)])
    def test_trace(self, matrix, trace):
        cupy_array = CuPyDense(matrix)
        assert cupy_array.trace() == trace


def test_no_checks_constr():
    """Checks if CuPyDense canbe properly inherited"""

    class wrapcpd(CuPyDense):
        pass
    wcpd1 = wrapcpd([[1,2],[0,1]])

    wcpd2 = wcpd1.to_array().transpose()

    np.testing.assert_array_equal(wcpd1.transpose().to_array(),wcpd2)
