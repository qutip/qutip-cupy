""" The qutip-cupy package provides a CuPy-based data layer for QuTiP. """
from qutip.core import data

from .version import version as __version__
from . import dense as cd

CuPyDense = cd.CuPyDense

data.to.add_conversions([
     (CuPyDense, data.Dense, cd.cupydense_from_dense),
     (data.Dense, CuPyDense, cd.dense_from_cupydense),
 ])
data.to.register_aliases(['cupyd'], CuPyDense)

data.adjoint.add_specialisations([
      (CuPyDense, CuPyDense, cd.cpd_adjoint),
     ])
data.transpose.add_specialisations([
      (CuPyDense, CuPyDense, cd.cpd_transpose),
     ])
data.conj.add_specialisations([
      (CuPyDense, CuPyDense, cd.cpd_conj),
     ])


# We must register the functions to the data layer but do not want
# the data layer to be callable from qutip_cupy
del data

del cd
