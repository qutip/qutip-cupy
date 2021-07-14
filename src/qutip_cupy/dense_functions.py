# Contains functions for dense_cupy, this are the same functions that 
# are defined ouside the dense.pyx file

def expect_dense(op, state):
    """
    Get the expectation value of the operator `op` over the state `state`.  The
    state can be either a ket or a density matrix.

    The expectation of a state is defined as the operation:
        state.adjoint() @ op @ state
    and of a density matrix:
        tr(op @ state)
    """
    # if state.shape[1] == 1:
    #     return _expect_dense_ket(op, state)
    # return _expect_dense_dense_dm(op, state)
    pass


def _expect_dense_ket(op, state):
    # _check_shape_ket(op, state)
    # cdef double complex out=0, sum
    # cdef size_t row, col, op_row_stride, op_col_stride
    # op_row_stride = 1 if op.fortran else op.shape[1]
    # op_col_stride = op.shape[0] if op.fortran else 1

    # for row in range(op.shape[0]):
    #     sum = 0
    #     for col in range(op.shape[0]):
    #         sum += (op.data[row * op_row_stride + col * op_col_stride] *
    #                 state.data[col])
    #     out += sum * conj(state.data[row])
    # return out
    pass


def _expect_dense_dense_dm(op, state):
    # _check_shape_dm(op, state)
    # cdef double complex out=0
    # cdef size_t row, col, op_row_stride, op_col_stride
    # cdef size_t state_row_stride, state_col_stride
    # state_row_stride = 1 if state.fortran else state.shape[1]
    # state_col_stride = state.shape[0] if state.fortran else 1
    # op_row_stride = 1 if op.fortran else op.shape[1]
    # op_col_stride = op.shape[0] if op.fortran else 1

    # for row in range(op.shape[0]):
    #     for col in range(op.shape[1]):
    #         out += op.data[row * op_row_stride + col * op_col_stride] * \
    #                state.data[col * state_row_stride + row * state_col_stride]
    # return out
    pass


def tidyup_dense(matrix, tol, inplace=True):
    # cdef Dense out = matrix if inplace else matrix.copy()
    # cdef double complex value
    # cdef size_t ptr
    # for ptr in range(matrix.shape[0] * matrix.shape[1]):
    #     value = matrix.data[ptr]
    #     if fabs(value.real) < tol:
    #         matrix.data[ptr].real = 0
    #     if fabs(value.imag) < tol:
    #         matrix.data[ptr].imag = 0
    # return out
    pass
