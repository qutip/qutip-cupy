# This is a dummy test file; delete it once the package actually has tests.


def test_import():
    import qutip_cupy
    assert qutip_cupy.__version__


def test_import_cupy():
    import cupy
    assert cupy.__version__


def test_norm_cupy():
    import cupy as cp
    import numpy as np
    x = cp.array([1, 2, 3])
    y = cp.array([2, 3, 4])
    z = x + y
    assert z == np.array([3, 5, 7])
