#cython: language_level=3

# Package-level relative imports in Cython (0.29.17) are temperamental.
from qutip_cupy cimport cupy_dense
from qutip_cupy.cupy_dense cimport Dense