import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
from pymanopt.manifolds.euclidean import Symmetric, Euclidean

from pyCovariance.manifolds.product import Product
from pyCovariance.testing import assert_allclose


def test_product():
    rnd.seed(123)

    p1 = 10
    p2 = 8
    p3 = 3
    w = (2, 3)
    m = Product([Symmetric(p1), Euclidean(p2, p3)], w)

    # rand
    X = m.rand()
    assert len(X) == 2
    assert X[0].shape == (p1, p1)
    assert X[1].shape == (p2, p3)
    assert_allclose(X[0], 1/2 * (X[0] + X[0].T))

    # randvec
    xi = m.randvec(X)
    assert len(xi) == 2
    assert xi[0].shape == (p1, p1)
    assert xi[1].shape == (p2, p3)
    assert_allclose(xi[0], 1/2 * (xi[0] + xi[0].T))

    # zerovec
    xi = m.zerovec(X)
    assert len(xi) == 2
    assert_allclose(xi[0], np.zeros((p1, p1)))
    assert_allclose(xi[1], np.zeros((p2, p3)))

    # inner
    xi = m.randvec(X)
    eta = m.randvec(X)
    res = w[0] * np.trace(xi[0].T@eta[0]) + w[1] * np.trace(xi[1].T@eta[1])
    assert_allclose(m.inner(X, xi, eta), res)

    # norm
    res = w[0] * np.trace(xi[0].T@xi[0]) + w[1] * np.trace(xi[1].T@xi[1])
    res = np.sqrt(res)
    assert_allclose(m.norm(X, xi), res)

    # dist
    X1 = m.rand()
    X2 = m.rand()
    d = w[0]*(la.norm(X1[0]-X2[0])**2) + w[1]*(la.norm(X1[1]-X2[1])**2)
    d = np.sqrt(d)
    assert_allclose(m.dist(X1, X2), d)

    # proj
    xi = [rnd.randn(p1, p1), rnd.randn(p2, p3)]
    p = m.proj(X, xi)
    assert len(p) == 2
    assert_allclose(p[0], 1/2 * (xi[0] + xi[0].T))
    assert_allclose(p[1], xi[1])

    # egrad2rgrad
    xi = [rnd.randn(p1, p1), rnd.randn(p2, p3)]
    p = m.proj(X, xi)
    res = [(1/w[0])*p[0], (1/w[1])*p[1]]
    grad = m.egrad2rgrad(X, xi)
    assert_allclose(grad[0], res[0])
    assert_allclose(grad[1], res[1])

    # ehess2rhess
    egrad = [rnd.randn(p1, p1), rnd.randn(p2, p3)]
    ehess = [rnd.randn(p1, p1), rnd.randn(p2, p3)]
    xi = [rnd.randn(p1, p1), rnd.randn(p2, p3)]
    p = m.proj(X, ehess)
    res = [(1/w[0])*p[0], (1/w[1])*p[1]]
    hess = m.ehess2rhess(X, egrad, ehess, xi)
    assert_allclose(hess[0], res[0])
    assert_allclose(hess[1], res[1])

    # exp
    X = m.rand()
    xi = m.randvec(X)
    res = [X[0] + xi[0], X[1] + xi[1]]
    exp = m.exp(X, xi)
    assert_allclose(exp[0], res[0])
    assert_allclose(exp[1], res[1])

    # retr
    X = m.rand()
    xi = m.randvec(X)
    res = [X[0] + xi[0], X[1] + xi[1]]
    retr = m.retr(X, xi)
    assert_allclose(retr[0], res[0])
    assert_allclose(retr[1], res[1])

    # log
    X1 = m.rand()
    X2 = m.rand()
    log = m.log(X1, X2)
    res = [X2[0] - X1[0], X2[1] - X1[1]]
    assert_allclose(log[0], res[0])
    assert_allclose(log[1], res[1])

    # transp
    xi = m.randvec(X1)
    transp = m.transp(X1, X2, xi)
    assert_allclose(transp[0], xi[0])
    assert_allclose(transp[1], xi[1])

    # pairmean
    X1 = m.rand()
    X2 = m.rand()
    pairmean = m.pairmean(X1, X2)
    res = [(X1[0] + X2[0]) / 2, (X1[1] + X2[1]) / 2]
    assert_allclose(pairmean[0], res[0])
    assert_allclose(pairmean[1], res[1])
