# This is a dummy test file; delete it once the package actually has tests.


def test_import():
    import qutip_cupy
    assert qutip_cupy.__version__


def test_class_dense():
    from qutip_cupy import Dense

    dense1 = Dense([[1,2,3,4,5]])
    assert len(dense1.shape) == 2

