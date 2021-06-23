from .version import version as __version__
from .cupy_dense import *


from qutip.core import data
data.to.add_conversions([
     (CuPyDense, data.Dense, cupydense_from_dense),
     (data.Dense, CuPyDense, dense_from_cupydense),
 ])
data.to.register_aliases(['cupyd'], CuPyDense)

data.adjoint.add_specialisations([
      (CuPyDense, CuPyDense, cpd_adjoint),
     ])
data.transpose.add_specialisations([
      (CuPyDense, CuPyDense, cpd_transpose),
     ])
data.conj.add_specialisations([
      (CuPyDense, CuPyDense, cpd_conj),
     ])


# We must register the functions to the data layer but do not want 
# the data layer to be callable from qutip_cupy
del data
