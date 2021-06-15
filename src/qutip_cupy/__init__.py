from .version import version as __version__

from qutip.core import data
from .cupy_dense import NewDataType

def dense_from_new(newarray):
    
    dense_np = data.Dense(newarray._cp.tolist())
    return dense_np

def new_from_dense(dense):
    
    dense_cp = NewDataType(dense.as_ndarray())
    return dense_cp


    
data.to.add_conversions([
     (NewDataType, data.Dense, new_from_dense),
     (data.Dense, NewDataType, dense_from_new),
 ])
