import cupy as cp


def _eigs_dense(data, isherm, vecs, eigvals, num_large, num_small):
    """
    Internal functions for computing eigenvalues and eigenstates for a dense
    matrix.
    """
    N = data.shape[0]
    kwargs = {}
    if eigvals != 0 and isherm:
        kwargs["eigvals"] = [0, num_small - 1] if num_small else [N - num_large, N - 1]
    if vecs:
        driver = cp. if isherm else cp.linalg.eigh
        evals, evecs = driver(data, **kwargs)
    else:
        driver = cp.linalg.eigvalsh if isherm else cp.linalg.eigvals
        evals = driver(data, **kwargs)
        evecs = None

    _zipped = list(zip(evals, range(len(evals))))
    _zipped.sort()
    evals, perm = list(zip(*_zipped))

    if vecs:
        evecs = np.array([evecs[:, k] for k in perm]).T

    if not isherm and eigvals > 0:
        if vecs:
            if num_small > 0:
                evals, evecs = evals[:num_small], evecs[:num_small]
            elif num_large > 0:
                evals, evecs = evals[(N - num_large) :], evecs[(N - num_large) :]
        else:
            if num_small > 0:
                evals = evals[:num_small]
            elif num_large > 0:
                evals = evals[(N - num_large) :]
    return np.array(evals), evecs


def eigs_dense(data, isherm=None, vecs=True, sort="low", eigvals=0):
    """
    Return eigenvalues and eigenvectors for a Dense matrix.  Takes no special
    keyword arguments; see the primary documentation in :func:`.eigs`.
    """
    if not isinstance(data, Dense):
        raise TypeError("expected data in Dense format but got " + str(type(data)))
    _eigs_check_shape(data)
    eigvals, num_large, num_small = _eigs_fix_eigvals(data, eigvals, sort)
    isherm = isherm if isherm is not None else _isherm(data)
    evals, evecs = _eigs_dense(
        data.as_ndarray(), isherm, vecs, eigvals, num_large, num_small
    )
    if sort == "high":
        # Flip arrays around.
        if vecs:
            evecs = np.fliplr(evecs)
        evals = evals[::-1]
    return (evals, Dense(evecs, copy=False)) if vecs else evals
