# This is a dummy test file; delete it once the package actually has tests.


def test_import():
    import qutip_cupy
    assert qutip_cupy.__version__

def test_conversion_cycle():
    from qutip_cupy import CuPyDense
    from qutip_cupy import data

    old_dense = data.Dense([[0,1,2,6,7,8]])

    tr1 = data.to[CuPyDense, data.Dense](old_dense)

    assert (old_dense.to_array() == data.to[data.Dense, CuPyDense](tr1).to_array()).all()


def test_shape():
    from qutip_cupy import CuPyDense
    from qutip_cupy import data

    cupy_dense = CuPyDense([[0,1,2,6,7,8]])

    assert (cupy_dense.shape == (1,6))





