from qutip_cupy import CuPyDense
import numpy as np
import pytest


def test_import():
    import qutip_cupy
    assert qutip_cupy.__version__


@pytest.mark.parametrize("shape", ((1, 2), (5, 10), (7, 3), (2, 5)))
def test_conversion_cycle(shape):

    from qutip.core import data

    qutip_dense = data.Dense(np.random.uniform(size=shape))

    tr1 = data.to(CuPyDense, qutip_dense)
    tr2 =  data.to(data.Dense, tr1)

    assert (qutip_dense.to_array() == tr2.to_array()).all()

@pytest.mark.parametrize("shape", ((1, 2), (5, 10), (7, 3), (2, 5)))
def test_shape(shape):

     cupy_dense = CuPyDense(np.random.uniform(size=shape))

     assert (cupy_dense.shape == shape)

@pytest.mark.parametrize("shape", ((1, 2), (5, 10), (7, 3), (2, 5)))
def test_adjoint(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_adj = CuPyDense(array).adjoint()
    qtpdense_adj = data.Dense(array).adjoint()

    assert (cpdense_adj.to_array() == qtpdense_adj.to_array()).all()



@pytest.mark.parametrize(["matrix",'trace'], [pytest.param([[0,1],[1,0]], 0),
                                              pytest.param([[2.j,1],[1,1]], 1+2.j)])
def test_trace(matrix, trace):

    cupy_array = CuPyDense(matrix)

    assert cupy_array.trace() == trace
