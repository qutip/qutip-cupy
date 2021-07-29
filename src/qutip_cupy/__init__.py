""" The qutip-cupy package provides a CuPy-based data layer for QuTiP. """

# we need to silence this specific warning
# remember to remove once QuTiP moves matplotlib
# to an official optional dependency
import warnings

try:
    __import__("cupy")
except ModuleNotFoundError:
    raise RuntimeError(
        "qutip_cupy requires cupy to be installed, please install cupy by following "
        "the instructions at https://docs.cupy.dev/en/stable/install.html"
    )

with warnings.catch_warnings():
    warnings.filterwarnings(
        action="ignore", category=UserWarning, message=r"matplotlib not found:"
    )

    from qutip.core import data

# qutip_cupy imports need to be after the cupy import check above
from .version import version as __version__  # noqa: E402
from . import dense as cd  # noqa: E402
from . import dense_functions as cdf  # noqa: E402

__all__ = ["__version__", "CuPyDense"]

CuPyDense = cd.CuPyDense

data.to.add_conversions(
    [
        (CuPyDense, data.Dense, cd.cupydense_from_dense),
        (data.Dense, CuPyDense, cd.dense_from_cupydense),
    ]
)
data.to.register_aliases(["cupyd"], CuPyDense)

data.adjoint.add_specialisations([(CuPyDense, CuPyDense, cd.adjoint_cupydense)])
data.transpose.add_specialisations([(CuPyDense, CuPyDense, cd.transpose_cupydense)])
data.conj.add_specialisations([(CuPyDense, CuPyDense, cd.conj_cupydense)])
data.trace.add_specialisations([(CuPyDense, cd.trace_cupydense)])
data.mul.add_specialisations([(CuPyDense, CuPyDense, cd.mul_cupydense)])
data.imul.add_specialisations([(CuPyDense, CuPyDense, cd.imul_cupydense)])
data.neg.add_specialisations([(CuPyDense, CuPyDense, cd.neg_cupydense)])
data.matmul.add_specialisations(
    [(CuPyDense, CuPyDense, CuPyDense, cd.matmul_cupydense)]
)
data.add.add_specialisations([(CuPyDense, CuPyDense, CuPyDense, cd.add_cupydense)])
data.sub.add_specialisations([(CuPyDense, CuPyDense, CuPyDense, cd.sub_cupydense)])
# constructor
data.diag.add_specialisations([(CuPyDense, cd.diags)])
data.identity.add_specialisations([(CuPyDense, cd.identity)])
data.zeros.add_specialisations([(CuPyDense, cd.zeros)])
# dense_functions
data.tidyup.add_specialisations([(CuPyDense, cdf.tidyup_dense)])
data.trace.add_specialisations([(CuPyDense, cdf.trace_cupydense)])
data.reshape.add_specialisations([(CuPyDense, CuPyDense, cdf.reshape_cupydense)])

# We must register the functions to the data layer but do not want
# the data layer or qutip_cupy.dense to be accessible from qutip_cupy
del data
del cd
