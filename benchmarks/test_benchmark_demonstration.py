#Remove this file after adding actual benchmarks

import pytest
import numpy as np

from qutip_cupy import CuPyDense

from .cpu_gpu_times_wrapper import GpuWrapper


@pytest.mark.benchmark()
def test_true_div(shape, benchmark):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    def divide_by_2(cp_arr):
        return cp_arr /2.
    cup_arr = CuPyDense(array)

    benchmark2 = GpuWrapper(benchmark)
    cpdense_tr = benchmark2.pedanticupy(divide_by_2, cup_arr)
    qtpdense_tr = data.Dense(array) /2.

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()
