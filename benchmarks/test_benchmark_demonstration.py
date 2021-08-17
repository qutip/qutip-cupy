# Remove this file after adding actual benchmarks
import pytest
import numpy as np
import cupy as cp
from qutip.core.data import Dense

from qutip_cupy import CuPyDense

import benchmark_tools
from benchmark_tools.cpu_gpu_times_wrapper import GpuWrapper

# from benchmarks import cpu_gpu_times_wrapper
# from cpgtw import GpuWrapper

# Set device_id
cp.cuda.Device(benchmark_tools._DEVICE).use()

# Supported dtypes
dtype_list = [CuPyDense, Dense]
dtype_ids = ["CuPy", "qutip(Dense)"]


@pytest.fixture(params=dtype_list, ids=dtype_ids)
def dtype(request):
    return request.param


@pytest.fixture(scope="function", params=[50, 100, 1000])
def size(request):
    return request.param


@pytest.mark.benchmark()
def test_matmul(size, benchmark, request, dtype):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id  # noqa:F821
    group = group.split("-")
    benchmark.group = "-".join(group[1:])
    benchmark.extra_info["dtype"] = "complex"  # group[0]

    array = np.random.uniform(size=(size, size)) + 1.0j * np.random.uniform(
        size=(size, size)
    )

    if dtype == CuPyDense:
        arr = CuPyDense(array)

    elif dtype == Dense:
        arr = Dense(array)

    def matmul_(arr):
        return arr @ arr

    benchmark2 = GpuWrapper(benchmark)
    cp_mult = benchmark2.pedanticupy(matmul_, (arr,))

    np_mult = matmul_(array)

    np.testing.assert_array_almost_equal(cp.asnumpy(cp_mult), np_mult)
