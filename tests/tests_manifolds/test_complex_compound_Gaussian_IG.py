import autograd.numpy as np
from autograd.numpy import linalg as la
from numpy import random as rnd, testing as np_testing

from pyCovariance.manifolds import ComplexCompoundGaussianIG
from pymanopt.tools.multi import multiherm
from pymanopt.manifolds.product import _ProductTangentVector


class TestSingleComplexCompoundGaussianIGManifold():
    def __init__(self):
        self._p = p = 15
        self._n = n = 100
        self.man = ComplexCompoundGaussianIG(p=p, n=n)

    def check_type_dim(self, x):
        p, n = self._p, self._n
        assert len(x) == 3
        assert x[0].dtype == np.complex128
        assert x[0].shape == (p, 1)
        assert x[1].dtype == np.complex128
        assert x[1].shape == (p, p)
        assert x[2].dtype == np.float64
        assert x[2].shape == (n, 1)

    def check_man(self, x):
        assert type(x) == list
        self.check_type_dim(x)

        # check symmetry
        np_testing.assert_allclose(x[1], multiherm(x[1]))

        # check positivity
        assert (la.eigvalsh(x[1]) > 0).all()
        assert (x[2] > 0).all()

    def check_tangent(self, x, u):
        self.check_man(x)
        assert type(u) == _ProductTangentVector
        self.check_type_dim(u)

        # check symmetry
        np_testing.assert_allclose(x[1], multiherm(x[1]))
        # check null trace
        np_testing.assert_almost_equal(np.trace(la.inv(x[1])@u[1]), 0)

    def test_dim(self):
        man = self.man
        p, n = self._p, self._n
        dim = 2*p + p*p + n
        np_testing.assert_equal(man.dim, dim)

    def test_rand(self):
        rnd.seed(123)

        x = self.man.rand()

        self.check_man(x)

    def test_randvec(self):
        rnd.seed(123)

        # Just test that randvec returns an element of the tangent space
        # with norm 1
        man = self.man
        x = man.rand()
        u = man.randvec(x)

        self.check_tangent(x, u)
        np_testing.assert_almost_equal(self.man.norm(x, u), 1)

    def test_zerovec(self):
        rnd.seed(123)

        p, n = self._p, self._n
        man = self.man
        x = man.rand()
        u = man.zerovec(x)

        self.check_tangent(x, u)
        assert (u[0] == np.zeros((p, 1))).all()
        assert (u[1] == np.zeros((p, p))).all()
        assert (u[2] == np.zeros((n, 1))).all()

    def test_inner(self):
        rnd.seed(123)

        n, p = self._n, self._p
        man = self.man
        x = man.rand()
        u = man.randvec(x)
        v = man.randvec(x)

        sigma_inv = la.inv(x[1])

        inner = 2 * np.sum(1/x[2]) * np.real(u[0].conj().T@sigma_inv@v[0])
        inner += n * np.real(np.trace(sigma_inv@u[1]@sigma_inv@v[1]))
        inner += p * (u[2]*(x[2]**(-1))).T@(v[2]*(x[2]**(-1)))
        np_testing.assert_almost_equal(inner, man.inner(x, u, v))
        assert type(man.inner(x, u, v)) == np.float64

    def test_norm(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        u = man.randvec(x)
        norm = np.abs(rnd.normal(loc=2, scale=3))
        u = u * norm
        np_testing.assert_almost_equal(man.norm(x, u), norm)

        assert type(man.norm(x, u)) == np.float64

    def test_proj(self):
        rnd.seed(123)

        n, p = self._n, self._p
        man = self.man
        x = man.rand()
        u = list()
        u.append(rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1)))
        u.append(rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p)))
        u.append(rnd.normal(size=(n, 1)))
        u = man.proj(x, u)

        self.check_tangent(x, u)
        np_testing.assert_allclose(u[0], man.proj(x, u)[0])
        np_testing.assert_allclose(u[1], man.proj(x, u)[1])
        np_testing.assert_allclose(u[2], man.proj(x, u)[2])

    def test_egrad2rgrad(self):
        rnd.seed(123)

        n, p = self._n, self._p
        man = self.man
        x = man.rand()
        egrad = list()
        egrad.append(rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1)))
        egrad.append(rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p)))
        egrad.append(rnd.normal(size=(n, 1)))
        grad = man.egrad2rgrad(x, egrad)
        self.check_tangent(x, grad)

    # def test_ehess2rhess(self):
    #     n = self.n
    #     x = self.man.rand()
    #     u = self.man.randvec(x)
    #     egrad = rnd.randn(n, n)
    #     ehess = rnd.randn(n, n)
    #     hess = self.man.ehess2rhess(x, egrad, ehess, u)
    #     hess_proj = self.man.proj(x, hess)

    #     np_testing.assert_allclose(hess, hess_proj)

    def test_retr(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()

        # retr(x, 0) = x
        u = man.zerovec(x)
        r = man.retr(x, u)
        np_testing.assert_allclose(r[0], x[0])
        np_testing.assert_allclose(r[1], x[1])
        np_testing.assert_allclose(r[2], x[2])

        # retr(x, u) = x + u.
        u = man.randvec(x)
        u = u * 1e-5
        r = man.retr(x, u)
        np_testing.assert_allclose(r[0], x[0] + u[0])
        np_testing.assert_allclose(r[1], x[1] + u[1])
        np_testing.assert_allclose(r[2], x[2] + u[2])

        self.check_man(r)

    def test_transp(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.randvec(x)
        t_u = man.transp(x, y, u)

        self.check_tangent(y, t_u)
