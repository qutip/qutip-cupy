# This is a dummy test file; delete it once the package actually has tests.


def test_import():
    import qutip_cupy
    assert qutip_cupy.__version__


def test_cupy_import():
    import cupy as cp
    pauli_x = cp.array([[0., 1.],
                        [1., 0.]], dtype=cp.complex64)

    pauli_y = cp.array([[0., -1.j],
                        [1.j, 0]], dtype=cp.complex64)

    commutator_x_z = pauli_x @ pauli_y - (pauli_y@pauli_x)

    _2j_pauli_z = cp.array([[2.j, 0.j],
                           [0.j, -2.j]], dtype=cp.complex64)

    assert (commutator_x_z == _2j_pauli_z).all()
