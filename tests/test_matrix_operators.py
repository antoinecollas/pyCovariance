import autograd.numpy as np
import autograd.numpy.random as rnd
import scipy.linalg as la

from pyCovariance.matrix_operators import\
        coshm, expm, g_invm, invsqrtm,\
        logm, powm, sinhm, sqrtm
from pyCovariance.matrix_operators import multitrace
from pyCovariance.testing import assert_allclose


def test_coshm():
    C = 2*np.eye(3)
    Ctrue = np.cosh(2)*np.eye(3)
    assert_allclose(coshm(C), Ctrue)


def test_expm():
    rnd.seed(123)

    # tests with a single matrix
    # test 1
    C = 2*np.eye(3)
    Ctrue = np.exp(2)*np.eye(3)
    assert_allclose(expm(C), Ctrue)

    # test 2
    Q, _ = la.qr(rnd.normal(size=(3, 3)) + 1j * rnd.normal(size=(3, 3)))
    D = rnd.normal(size=(3))
    A = Q@np.diag(D)@Q.conj().T
    exp_A = Q@np.diag(np.exp(D))@Q.conj().T
    assert la.norm(expm(A) - exp_A) / la.norm(exp_A) < 0.01

    # test 3
    approx = np.eye(3) + A + A@A/2 + A@A@A/6 + A@A@A@A/24
    assert la.norm(expm(A) - approx) / la.norm(approx) < 0.01

    # test on a batch of matrices
    A = np.zeros((10, 3, 3), dtype=np.complex128)
    exp_A = np.zeros((len(A), 3, 3), dtype=np.complex128)

    for i in range(len(A)):
        Q, _ = la.qr(rnd.normal(size=(3, 3)) + 1j * rnd.normal(size=(3, 3)))
        D = rnd.normal(size=(3))
        A[i] = Q@np.diag(D)@Q.conj().T
        exp_A[i] = Q@np.diag(np.exp(D))@Q.conj().T

    assert expm(A).dtype == np.complex128
    assert expm(A).shape == (len(A), 3, 3)

    for i in range(len(A)):
        assert la.norm(expm(A)[i] - exp_A[i]) / la.norm(exp_A[i]) < 0.01


def test_g_invm():
    C = np.eye(3) - np.eye(3)
    C[0, 0] = 3
    C[1, 1] = 2
    Ctrue = np.eye(3) - np.eye(3)
    Ctrue[0, 0] = 1/3
    Ctrue[1, 1] = 1/2
    assert_allclose(g_invm(C), Ctrue)


def test_invsqrtm():
    C = 2*np.eye(3)
    Ctrue = (1.0/np.sqrt(2))*np.eye(3)
    assert_allclose(invsqrtm(C), Ctrue)


def test_logm():
    C = 2*np.eye(3)
    Ctrue = np.log(2)*np.eye(3)
    assert_allclose(logm(C), Ctrue)


def test_powm():
    C = 2*np.eye(3)
    Ctrue = (2**-1)*np.eye(3)
    assert_allclose(powm(C, -1), Ctrue)


def test_sinhm():
    C = 2*np.eye(3)
    Ctrue = np.sinh(2)*np.eye(3)
    assert_allclose(sinhm(C), Ctrue)


def test_sqrtm():
    C = 2*np.eye(3)
    Ctrue = np.sqrt(2)*np.eye(3)
    assert_allclose(sqrtm(C), Ctrue)


def test_multitrace():
    A = rnd.normal(size=(10, 10))
    assert_allclose(multitrace(A), np.trace(A))

    A = rnd.normal(size=(5, 10, 10))
    res = multitrace(A)
    assert len(res) == A.shape[0]
    for i in range(A.shape[0]):
        assert_allclose(res[i], np.trace(A[i, :, :]))
