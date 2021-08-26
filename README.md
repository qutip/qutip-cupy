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

- `conda install -c conda-forge cupy`

To install `QuTiP >5.0.0` while it is not yet released we recommend:

- `python -mpip install git+https://github.com/qutip/qutip.git@dev.major`

Now you can safely install `qutip_cupy`

- `python -mpip install git+https://github.com/qutip/qutip-cupy.git`

Usage
------------

The main object that `qutip-cupy` provides is `CuPyDense` which is a `CuPy` based interface to store `Qobj`'s data.

When working with a new `Qobj` you may proceed as follows:

``` python
import qutip 
import qutip_cupy

qobj = qutip.Qobj([0, 1], dtype="cupyd")
qobj.data

```

This then returns

``` python
<qutip_cupy.dense.CuPyDense at 0x7fea2b2338c0>

```

In this way you can create CuPyDense arrays that live in the defult GPU device on your environment. If you have more than one GPU we recommend that you check the documentation if you want to choose a custom one. We also provide some custom constructors to initialize `CuPyDense` arrays.

Operations that return an array will return control inmediately to the user, while scalar valued functions will block and return the result to general memory.

You can operate a CuPyDense-backed state with a CuPyDense-backed unitary, and the result will also be CuPyDense-backed.

``` python
import numpy as np 
theta = (1/2)*np.pi

U = qutip.Qobj([[np.cos(theta), 1.j*np.sin(theta)],[-1.j*np.sin(theta),np.cos(theta) ]]).to('cupyd')

qobj_end = U @ qobj

qobj_end.data

```

``` python
<qutip_cupy.dense.CuPyDense at 0x7f1190688d20>

```

You can then calculate the overlap of  the new state with the original state. The resulting overlap lives in the CPU and if you wanted to then calculate the probability of finding the new state to be the original state (i.e. you were to project on a suitable base that has as an element the original state) one should use CPU-bound computation, in this case we call `np.linalg.norm` .

``` python

overlap = qobj_end.overlap(qobj)
np.linalg.norm(overlap)

```

``` python
6.123233995736766e-17
```

You can now start working with `CuPy` based arrays seamlessly. `qutip-cupy` takes care to dispatch all functions to specialisations on `CuPyDense` arrays, and if there is no specialisation for the given function yet `QuTiP`'s data-layer will force a conversion to one of its own data-types and run the required function within the CPU. We recommend that you check our `GitHub` issues to stay up to date on any missing or new specialisations.

Benchmarks
------------

This is a work in progress.
