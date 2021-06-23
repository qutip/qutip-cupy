from .version import version as __version__

from .cupy_dense import CuPyDense, cupydense_from_dense, dense_from_cupydense

from qutip.core import data
data.to.add_conversions([
     (CuPyDense, data.Dense, cupydense_from_dense),
     (data.Dense, CuPyDense, dense_from_cupydense),
 ])
# We must register the functions to the data layer but do not want 
# the data layer to be callable from qutip_cupy
del data
