import time
import pytest
import qutip
import numpy as np
import qutip_cupy



def test_import2():
    import qutip_cupy
    assert qutip_cupy.__version__


def test_matmul():
    pass