import autograd.numpy as np

def vec(mat):
    return mat.ravel('F')


def vech(mat):
    # Gets Fortran-order
    rows, cols = np.triu_indices(len(mat))
    vec = mat[rows, cols]
    return vec


def _diag_indices(n):
    rows, cols = np.diag_indices(n)
    return rows * n + cols


def unvec(v):
    k = int(np.sqrt(len(v)))
    assert(k * k == len(v))
    return v.reshape((k, k), order='F')


def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows), dtype=v.dtype)
    result[np.triu_indices(rows)] = v
    result = result + result.conj().T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result
