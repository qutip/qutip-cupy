# Remove this file after adding actual benchmarks

import pytest
import numpy as np
import cupy as cp
import qutip as qt
from qutip_cupy import CuPyDense

# from qutip_cupy import CuPyDense

from .cpu_gpu_times_wrapper import GpuWrapper

# Supported dtypes
dtype_list = [np, CuPyDense, qt.data.Dense, qt.data.CSR]
dtype_ids = ["numpy", "CuPy", "qutip(Dense)", "qutip(CSR)"]


@pytest.fixture(params=dtype_list, ids=dtype_ids)
def dtype(request):
    return request.param


@pytest.fixture(scope="function", params=((1000, 1000)))  # ,(2000, 2000)))
def shape(request):
    return request.param


@pytest.mark.benchmark()
def test_matmul(shape, benchmark, request):
    # Group benchmark by operation, density and size.
    group = request.node.callspec.id  # noqa:F821
    group = group.split("-")
    benchmark.group = "-".join(group[1:])
    benchmark.extra_info["dtype"] = "complex"  # group[0]

    array = np.random.uniform(size=shape) + 1.0j * np.random.uniform(size=shape)

    cp_arr = cp.array(array)

    def matmul_(cp_arr):
        return cp_arr @ cp_arr

    benchmark2 = GpuWrapper(benchmark)
    cp_mult = benchmark2.pedanticupy(matmul_, (cp_arr,))

    np_mult = matmul_(array)

    np.testing.assert_array_almost_equal(cp.asnumpy(cp_mult), np_mult)
