from qutip_cupy import CuPyDense
import numpy as np
import pytest

@pytest.fixture(scope="function", params=((1, 2), (5, 10), (7, 3), (2, 5)))
def shape(request):
    return request.param



def test_conversion_cycle(shape):

    from qutip.core import data

    qutip_dense = data.Dense(np.random.uniform(size=shape))

    tr1 = data.to(CuPyDense, qutip_dense)
    tr2 = data.to(data.Dense, tr1)

    assert (qutip_dense.to_array() == tr2.to_array()).all()


def test_shape(shape):

    cupy_dense = CuPyDense(np.random.uniform(size=shape))

    assert (cupy_dense.shape == shape)


def test_adjoint(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_adj = CuPyDense(array).adjoint()
    qtpdense_adj = data.Dense(array).adjoint()

    assert (cpdense_adj.to_array() == qtpdense_adj.to_array()).all()



@pytest.mark.parametrize(["matrix", "trace"], [pytest.param([[0, 1],[1, 0]], 0),
                                              pytest.param([[2.j, 1],[1, 1]], 1+2.j)])
def test_trace(matrix, trace):

    cupy_array = CuPyDense(matrix)

    assert cupy_array.trace() == trace





def test_true_div(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array)  /2.
    qtpdense_tr = data.Dense(array) /2.

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()


def test_itrue_div(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__itruediv__(2.)
    qtpdense_tr = data.Dense(array).__itruediv__(2.)

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()


def test_mul(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__mul__(2.+1.j)
    qtpdense_tr = data.Dense(array).__mul__(2.+1.j)

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()
