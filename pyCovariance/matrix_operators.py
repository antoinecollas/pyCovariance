# Inspired from: https://github.com/alexandrebarachant/pyRiemann/
# blob/master/pyriemann/utils/base.py
import autograd.numpy as np
import autograd.numpy.linalg as la


def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    eigvals, eigvects = la.eigh(Ci)
    if Ci.ndim == 2:
        eigvals = operator(eigvals)[np.newaxis, ...]
        A = eigvects*eigvals
        B = np.transpose(np.conjugate(eigvects))
        return A@B
    else:
        eigvals = operator(eigvals)[:, np.newaxis, ...]
        A = eigvects*eigvals
        B = np.transpose(np.conjugate(eigvects), axes=(0, 2, 1))
        C = np.einsum('ijk,ikl->ijl', A, B)
        return C


def sqrtm(Ci):
    """Return the matrix square root of a cov. matrix defined by:

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2}
            \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the cov. matrix
    :returns: the matrix square root

    """
    return _matrix_operator(Ci, np.sqrt)


def logm(Ci):
    """Return the matrix logarithm of a cov. matrix defined by:

    .. math::
            \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the cov. matrix
    :returns: the matrix logarithm

    """
    return _matrix_operator(Ci, np.log)


def expm(Ci):
    """Return the matrix exponential of a cov. matrix defined by:

    .. math::
            \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the cov. matrix
    :returns: the matrix exponential

    """
    return _matrix_operator(Ci, np.exp)


def coshm(Ci):
    """Return the matrix hyperbolic cosine of a cov. matrix defined by:

    .. math::
            \mathbf{C} = \mathbf{V} \cosh{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the cov. matrix
    :returns: the matrix hyperbolic cosine

    """
    return _matrix_operator(Ci, np.cosh)


def sinhm(Ci):
    """Return the matrix hyperbolic sine of a cov. matrix defined by:

    .. math::
            \mathbf{C} = \mathbf{V} \sinh{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the cov. matrix
    :returns: the matrix hyperbolic sine

    """
    return _matrix_operator(Ci, np.sinh)


def invsqrtm(Ci):
    """Return the inverse matrix square root of a cov. matrix defined by:

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2}
            \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the cov. matrix
    :returns: the inverse matrix square root

    """
    def isqrt(x):
        return 1. / np.sqrt(x)
    return _matrix_operator(Ci, isqrt)


def powm(Ci, alpha):
    """Return the matrix power :math:`\\alpha` of a cov. matrix defined by:

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha}
            \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the cov. matrix
    :param alpha: the power to apply
    :returns: the matrix power

    """
    def power(x):
        return x**alpha
    return _matrix_operator(Ci, power)


def g_invm(Ci):
    """Return the g inverse of a cov. matrix defined by:

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-}
            \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the cov. matrix
    :param alpha: the power to apply
    :returns: the matrix power

    """
    def g_inv(x):
        x[np.abs(x) > 1e-15] = 1./x[np.abs(x) > 1e-15]
        return x
    return _matrix_operator(Ci, g_inv)


# the following is inspired by pymanopt: multi.py

def multiprod(A, B):
    """
    Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    assumed to be arrays containing M matrices, that is, A and B have
    dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    in A with the corresponding matrix in B, using matrix multiplication.
    so multiprod(A, B) has dimensions (M, N, Q).
    """

    # First check if we have been given just one matrix
    if len(np.shape(A)) == 2:
        return np.dot(A, B)

    return np.einsum('ijk,ikl->ijl', A, B)


def multitrace(A):
    """
    A is assumed to be an array containing M matrices, that is,
    A has a dimension (M, N, N).
    multitrace computes the trace of each matrix in A.
    so multitrace(A) has dimension M.
    """
    # np.einsum('...ii', A) works whaterver the dimension of A.
    # Unfortunately, it is not differentiable using Autograd...
    if A.ndim == 3:
        A = np.transpose(A, (1, 2, 0))
    return np.trace(A)


def multitransp(A):
    """
    Inspired by MATLAB multitransp function by Paolo de Leva. A is assumed to
    be an array containing M matrices, each of which has dimension N x P.
    That is, A is an M x N x P array. Multitransp then returns an array
    containing the M matrix transposes of the matrices in A, each of which
    will be P x N.
    """
    # First check if we have been given just one matrix
    if A.ndim == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))


def multihconj(A):
    return np.conjugate(multitransp(A))


def multisym(A):
    # Inspired by MATLAB multisym function by Nicholas Boumal.
    return 0.5 * (A + multitransp(A))


def multiherm(A):
    # Inspired by MATLAB multiherm function by Nicholas Boumal.
    return 0.5 * (A + multihconj(A))


def multiskew(A):
    # Inspired by MATLAB multiskew function by Nicholas Boumal.
    return 0.5 * (A - multitransp(A))


def multieye(k, n):
    # Creates a k x n x n array containing k (n x n) identity matrices.
    return np.tile(np.eye(n), (k, 1, 1))
