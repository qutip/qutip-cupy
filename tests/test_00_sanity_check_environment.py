# This test is used to check if the environment is properly set.
# If there are no GPUs in the environment CuPy imports will break.

import warnings


def test_import():
    import qutip_cupy

    assert qutip_cupy.__version__


def test_import_cupy():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import cupy
    assert cupy.__version__


def test_add_cupy():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import cupy as cp
    import numpy as np

    x = cp.array([1, 2, 3])
    y = cp.array([2, 3, 4])
    z = x + y
    np.testing.assert_array_equal(cp.asnumpy(z), np.array([3, 5, 7]))
