qutip-cupy: CuPy backend for QuTiP
==================================

A plugin for [QuTiP](https://qutip.org) providing a [CuPy](https://cupy.dev) linear-algebra backend for GPU computation.

Support
-------

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat)](https://unitary.fund)
[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)

We are proud to be affiliated with [Unitary Fund](https://unitary.fund) and [NumFOCUS](https://numfocus.org).
QuTiP development is supported by [Nori's lab](https://dml.riken.jp/) at RIKEN, by the University of Sherbrooke, and by Aberystwyth University, [among other supporting organizations](https://qutip.org/#supporting-organizations).
Initial work on this project was sponsored by [Google Summer of Code 2021](https://summerofcode.withgoogle.com).

Installation
------------

`qutip-cupy` is not yet officially released.

If you want to try out the package you will need to have a CUDA enabled GPU, `QuTiP >5.0.0` and `CuPy`.
We recommend using a conda environment `Python >= 3.7`.
To install `CuPy` we recommend the following steps:

- conda install -c conda-forge cupy

To install `QuTiP >5.0.0` while it is not yet released we recommend:

- python -mpip install git+https://github.com/qutip/qutip.git@dev.major

Now you can safely install `qutip_cupy`

- python -mpip install git+https://github.com/qutip/qutip-cupy.git

Usage
------------

The main object that `qutip-cupy` provides is `CuPyDense` which is a `CuPy` based interface to store `Qobj`'s data.

When working with a new `Qobj` you may proceed as follows:

``` python
import qutip 
import qutip_cupy
from qutip_cupy import CuPyDense

qobj = qutip.Qobj( CuPyDense([2,1] ))
qobj.data
```

This then returns

``` python
<qutip_cupy.dense.CuPyDense at 0x7fea2b2338c0>

```

