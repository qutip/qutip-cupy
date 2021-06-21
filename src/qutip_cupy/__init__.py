from .version import version as __version__
# First-class type imports

#from src.qutip_cupy import cupy_dense
#from src.qutip_cupy.cupy_dense import Dense 

from .version import version as __version__

from qutip.core import data

from src.qutip_cupy import cupy_dense
from src.qutip_cupy.cupy_dense import CuPyDense

def dense_from_new(newarray):
    
    dense_np = data.Dense(newarray._cp.tolist())
    return dense_np

def new_from_dense(dense):
    
    dense_cp = CuPyDense(dense.as_ndarray())
    return dense_cp


    
data.to.add_conversions([
     (CuPyDense, data.Dense, new_from_dense),
     (data.Dense, CuPyDense, dense_from_new),
 ])
