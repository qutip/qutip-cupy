#Remove this file after adding actual benchmarks

import pytest
import numpy as np
import cupy as cp

#from qutip_cupy import CuPyDense

from .cpu_gpu_times_wrapper import GpuWrapper

@pytest.fixture(scope="function", params=((1000, 1000),(2000, 2000)))
def shape(request):
    return request.param

@pytest.mark.benchmark()
def test_matmul(shape, benchmark):

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cp_arr = cp.array(array)

    def matmul_(cp_arr):
        return array @ array

    benchmark2 = GpuWrapper(benchmark)
    cp_mult = benchmark2.pedanticupy(matmul_, (cp_arr,))

    print((cp_arr @ cp_arr).__class__)
    print(cp_arr.__class__)
    print(cp_mult.__class__)
    print(cp_mult.shape)
    np_mult = matmul_(array)

    np.testing.assert_array_almost_equal(cp_mult.asnumpy(), np_mult)
