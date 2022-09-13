import autograd
import autograd.numpy as np
from autograd.numpy import linalg as la
from autograd.scipy.special import logsumexp
from numpy import random as rnd
from pymanopt.manifolds.product import _ProductTangentVector

from pyCovariance.manifolds import\
        ComplexCompoundGaussianIGConstrainedScatter,\
        ComplexCompoundGaussianIGConstrainedTexture
from pyCovariance.matrix_operators import multihconj, multiherm, powm
from pyCovariance.testing import assert_allclose


class TestSingleComplexCompoundGaussianIGConstrainedScatterManifold():
    def __init__(self):
        self._p = p = 15
        self._n = n = 100
        self.man = ComplexCompoundGaussianIGConstrainedScatter(p=p, n=n)

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
        assert_allclose(x[1], multiherm(x[1]))

        # check positivity
        assert (la.eigvalsh(x[1]) > 0).all()
        assert (x[2] > 0).all()

        # check unit determinant
        assert_allclose(la.det(x[1]), 1)

    def check_tangent(self, x, u):
        self.check_man(x)
        assert type(u) == _ProductTangentVector
        self.check_type_dim(u)

        # check symmetry
        assert_allclose(x[1], multiherm(x[1]))
        # check null trace
        assert_allclose(np.trace(la.inv(x[1])@u[1]), 0, atol=1e-8)

    def test_dim(self):
        man = self.man
        p, n = self._p, self._n
        dim = 2*p + p*p - 1 + n
        assert man.dim == dim

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
        assert_allclose(self.man.norm(x, u), 1)

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
        assert_allclose(inner, man.inner(x, u, v))
        assert type(man.inner(x, u, v)) == np.float64

    def test_norm(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        u = man.randvec(x)
        norm = np.abs(rnd.normal(loc=2, scale=3))
        u = u * norm
        assert_allclose(man.norm(x, u), norm)

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
        assert_allclose(u[0], man.proj(x, u)[0])
        assert_allclose(u[1], man.proj(x, u)[1])
        assert_allclose(u[2], man.proj(x, u)[2])

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

    #     assert_allclose(hess, hess_proj)

    def test_retr(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()

        # retr(x, 0) = x
        u = man.zerovec(x)
        r = man.retr(x, u)
        assert_allclose(r[0], x[0])
        assert_allclose(r[1], x[1])
        assert_allclose(r[2], x[2])

        # retr(x, u) = x + u.
        u = man.randvec(x)
        u = u * 1e-5
        r = man.retr(x, u)
        assert_allclose(r[0], x[0] + u[0])
        assert_allclose(r[1], x[1] + u[1])
        assert_allclose(r[2], x[2] + u[2])

        self.check_man(r)

    def test_transp(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.randvec(x)
        t_u = man.transp(x, y, u)

        self.check_tangent(y, t_u)


class TestSingleComplexCompoundGaussianIGConstrainedTextureManifold():
    def __init__(self):
        self._p = p = 15
        self._n = n = 100
        self._alpha = alpha = 0.33
        self.man = ComplexCompoundGaussianIGConstrainedTexture(
            p=p, n=n, alpha=alpha)

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
        assert_allclose(x[1], multiherm(x[1]))

        # check positivity
        assert (la.eigvalsh(x[1]) > 0).all()
        assert (x[2] > 0).all()

        # check textures with unit prod
        assert_allclose(np.prod(x[2]), 1)

    def check_tangent(self, x, u):
        self.check_man(x)
        assert type(u) == _ProductTangentVector
        self.check_type_dim(u)

        # check symmetry
        assert_allclose(x[1], multiherm(x[1]))

        # check null trace
        assert_allclose(np.sum((x[2]**-1)*u[2]), 0)

    def test_dim(self):
        man = self.man
        p, n = self._p, self._n
        dim = 2*p + p*p - 1 + n
        assert man.dim == dim

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
        assert_allclose(self.man.norm(x, u), 1)

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

        assert type(man.inner(x, u, v)) == np.float64
        assert_allclose(man.inner(x, u, v), inner)

    def test_norm(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        u = man.randvec(x)
        norm = np.abs(rnd.normal(loc=2, scale=3))
        u = u * norm

        assert type(man.norm(x, u)) == np.float64
        assert_allclose(man.norm(x, u), norm)

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
        assert_allclose(u[0], man.proj(x, u)[0])
        assert_allclose(u[1], man.proj(x, u)[1])
        assert_allclose(u[2], man.proj(x, u)[2])

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

    def test_retr(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()

        # retr(x, 0) = x
        u = man.zerovec(x)
        r = man.retr(x, u)
        assert_allclose(r[0], x[0])
        assert_allclose(r[1], x[1])
        assert_allclose(r[2], x[2])

        # retr(x, u) = x + u.
        u = man.randvec(x)
        u = u * 1e-5
        r = man.retr(x, u)
        assert_allclose(r[0], x[0] + u[0])
        assert_allclose(r[1], x[1] + u[1])
        assert_allclose(r[2], x[2] + u[2])

        self.check_man(r)

    def test_transp(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.randvec(x)
        t_u = man.transp(x, y, u)

        self.check_tangent(y, t_u)

    def test_div_KL(self):
        rnd.seed(123)

        n, p = self._n, self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_KL(x1, x1)**2
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_KL(x1, x2)**2
        assert type(d) is np.float64
        assert d > 1e-2

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T, x1[2]]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T, x2[2]]
        assert_allclose(man.div_KL(x1, x2)**2, d)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_KL(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1, 2])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 3
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)
        assert egrad[2].shape == (n, 1)

        # case sigma_1 == sigma_2 and tau_1 == tau_2:
        # scaled Mahalanobis distance
        x2[1] = x1[1]
        x2[2] = x1[2]
        delta_mu = x2[0] - x1[0]
        s_inv = powm(x1[1], -1)
        d = delta_mu.conj().T @ s_inv @ delta_mu
        d = np.sum(1/x2[2]) * np.real(d)
        assert_allclose(man.div_KL(x1, x2)**2, d)

        # case mu_1 == mu_2
        x1 = man.rand()
        x2 = man.rand()
        x2[0] = x1[0]

        d = np.sum(x1[2]/x2[2]) * np.real(np.trace(powm(x2[1], -1)@x1[1]))
        d = d - n*p
        d = d + n*np.log(np.real(la.det(x2[1])))
        d = d - n*np.log(np.real(la.det(x1[1])))
        d = np.sqrt(d)
        assert_allclose(man.div_KL(x1, x2), d)

    def test_div_alpha(self):
        rnd.seed(123)

        n, p = self._n, self._p
        alpha = self._alpha

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability, positivity, symmetry
        d = man.div_alpha(x1, x1)**2
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_alpha(x1, x2)**2
        assert type(d) is np.float64
        assert d > 1e-2
        if alpha == 0.5:
            assert_allclose(man.div_alpha(x1, x2), man.div_alpha(x2, x1))

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T, x1[2]]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T, x2[2]]
        assert_allclose(man.div_alpha(x1, x2)**2, d)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_alpha(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1, 2])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 3
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)
        assert egrad[2].shape == (n, 1)

        # case sigma_1 == sigma_2 and tau_1 == tau_2 == 1:
        # Mahalanobis distance
        x2[1] = x1[1]
        x1[2] = np.ones_like(x1[2])
        x2[2] = np.ones_like(x2[2])
        delta_mu = x2[0] - x1[0]
        s_inv = la.inv(x1[1])
        d = np.real(delta_mu.conj().T@s_inv@delta_mu).reshape(())
        d = n*alpha*(1-alpha)*d
        assert_allclose(man.div_alpha(x1, x2)**2, d, rtol=1e-3)

        # case mu_1 == mu_2, sigma_1 == sigma_2
        x1 = man.rand()
        x2 = man.rand()
        x2[0] = x1[0]
        x2[1] = x1[1]
        t = (1-alpha) * x1[2] + alpha * x2[2]
        d = p*np.sum(np.log(t))
        assert_allclose(man.div_alpha(x1, x2)**2, d)

    def test_div_alpha_sym(self):
        rnd.seed(123)

        n, p = self._n, self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability, positivity, symmetry
        d = man.div_alpha_sym(x1, x1)**2
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_alpha_sym(x1, x2)**2
        assert type(d) is np.float64
        assert d > 1e-2

        # symmetry
        assert_allclose(man.div_alpha_sym(x1, x2), man.div_alpha_sym(x2, x1))

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T, x1[2]]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T, x2[2]]
        assert_allclose(man.div_alpha_sym(x1, x2)**2, d)

        # value
        d1 = man.div_alpha(x1, x2)**2
        d2 = man.div_alpha(x2, x1)**2
        assert_allclose(0.5*d1 + 0.5*d2, man.div_alpha_sym(x1, x2)**2)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_alpha_sym(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1, 2])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 3
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)
        assert egrad[2].shape == (n, 1)

    def test_div_alpha_real_case(self):
        rnd.seed(123)

        n, p = self._n, self._p
        alpha = self._alpha

        man = self.man
        x1 = man.rand()
        x1[0] = np.real(x1[0])
        x1[1] = np.real(x1[1])
        x2 = man.rand()
        x2[0] = np.real(x2[0])
        x2[1] = np.real(x2[1])

        # separability, positivity, symmetry
        d = man.div_alpha_real_case(x1, x1)**2
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_alpha_real_case(x1, x2)**2
        assert type(d) is np.float64
        assert d > 1e-2
        if alpha == 0.5:
            assert_allclose(
                man.div_alpha_real_case(x1, x2),
                man.div_alpha_real_case(x2, x1)
            )

        # invariance
        A = rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T, x1[2]]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T, x2[2]]
        assert_allclose(man.div_alpha_real_case(x1, x2)**2, d)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_alpha_real_case(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1, 2])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 3
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)
        assert egrad[2].shape == (n, 1)

        # case sigma_1 == sigma_2 and tau_1 == tau_2 == 1:
        # Mahalanobis distance
        x2[1] = x1[1]
        x1[2] = np.ones_like(x1[2])
        x2[2] = np.ones_like(x2[2])
        delta_mu = x2[0] - x1[0]
        s_inv = la.inv(x1[1])
        d = np.real(delta_mu.conj().T@s_inv@delta_mu).reshape(())
        d = 0.5*n*alpha*(1-alpha)*d
        assert_allclose(man.div_alpha_real_case(x1, x2)**2, d, rtol=1e-3)

        # case mu_1 == mu_2, sigma_1 == sigma_2
        x1 = man.rand()
        x2 = man.rand()
        x2[0] = x1[0]
        x2[1] = x1[1]
        t = (1-alpha) * x1[2] + alpha * x2[2]
        d = 0.5*p*np.sum(np.log(t))
        assert_allclose(man.div_alpha_real_case(x1, x2)**2, d)

        # real KL is 1/2 of complex KL
        x1 = man.rand()
        x2 = man.rand()
        KL_man = ComplexCompoundGaussianIGConstrainedTexture(
            p=p, n=n, k=1, alpha=0)
        temp0 = KL_man.div_alpha_real_case(x1, x2)**2
        temp1 = KL_man.div_alpha(x1, x2)**2
        assert_allclose(temp0, 0.5*temp1)

    def test_div_alpha_sym_real_case(self):
        rnd.seed(123)

        n, p = self._n, self._p

        man = self.man
        x1 = man.rand()
        x1[0] = x1[0].real
        x1[1] = x1[1].real
        x1[2] = x1[2].real
        x2 = man.rand()
        x2[0] = x2[0].real
        x2[1] = x2[1].real
        x2[2] = x2[2].real

        # separability, positivity, symmetry
        d = man.div_alpha_sym_real_case(x1, x1)**2
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_alpha_sym_real_case(x1, x2)**2
        assert type(d) is np.float64
        assert d > 1e-2

        # symmetry
        d1 = man.div_alpha_sym_real_case(x1, x2)
        d2 = man.div_alpha_sym_real_case(x2, x1)
        assert_allclose(d1, d2)

        # invariance
        A = rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.T, x1[2]]
        x2 = [A@x2[0] + mu, A@x2[1]@A.T, x2[2]]
        assert_allclose(man.div_alpha_sym_real_case(x1, x2)**2, d)

        # value
        d1 = man.div_alpha_real_case(x1, x2)**2
        d2 = man.div_alpha_real_case(x2, x1)**2
        tmp = 0.5*d1 + 0.5*d2
        assert_allclose(tmp, man.div_alpha_sym_real_case(x1, x2)**2)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_alpha_sym_real_case(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1, 2])(*theta))
        assert type(egrad) is list
        assert len(egrad) == 3
        assert egrad[0].shape == (p, 1)
        assert egrad[0].dtype == np.float64
        assert egrad[1].shape == (p, p)
        assert egrad[1].dtype == np.float64
        assert egrad[2].shape == (n, 1)
        assert egrad[2].dtype == np.float64


class TestProductComplexCompoundGaussianIGConstrainedTextureManifold():
    def __init__(self):
        self._k = k = 5
        self._p = p = 15
        self._n = n = 100
        self._alpha = alpha = 0.5
        self.man = ComplexCompoundGaussianIGConstrainedTexture(
            p=p, n=n, k=k, alpha=alpha)

    def check_type_dim(self, x):
        k, p, n = self._k, self._p, self._n
        assert len(x) == 3
        assert x[0].dtype == np.complex128
        assert x[0].shape == (k, p, 1)
        assert x[1].dtype == np.complex128
        assert x[1].shape == (k, p, p)
        assert x[2].dtype == np.float64
        assert x[2].shape == (k, n, 1)

    def check_man(self, x):
        k = self._k

        assert type(x) == list
        self.check_type_dim(x)

        # check symmetry
        assert_allclose(x[1], multiherm(x[1]))

        # check positivity
        for i in range(k):
            assert (la.eigvalsh(x[1][i, :, :]) > 0).all()
        assert (x[2] > 0).all()

        # check textures with unit prod
        for i in range(k):
            assert_allclose(np.prod(x[2][i, :, :]), 1)

    def check_tangent(self, x, u):
        k = self._k

        self.check_man(x)
        assert type(u) == _ProductTangentVector
        self.check_type_dim(u)

        # check symmetry
        assert_allclose(x[1], multiherm(x[1]))

        # check null trace
        for i in range(k):
            assert_allclose(np.sum((x[2][i, :, :]**-1)*u[2][i, :, :]), 0)

    def test_dim(self):
        man = self.man
        k, p, n = self._k, self._p, self._n
        dim = k*(2*p + p*p + n - 1)
        assert man.dim == dim

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
        assert_allclose(self.man.norm(x, u), 1)

    def test_zerovec(self):
        rnd.seed(123)

        k, n, p = self._k, self._n, self._p
        man = self.man
        x = man.rand()
        u = man.zerovec(x)

        self.check_tangent(x, u)
        assert (u[0] == np.zeros((k, p, 1))).all()
        assert (u[1] == np.zeros((k, p, p))).all()
        assert (u[2] == np.zeros((k, n, 1))).all()

    def test_inner(self):
        rnd.seed(123)

        k, n, p = self._k, self._n, self._p
        man = self.man
        x = man.rand()
        u = man.randvec(x)
        v = man.randvec(x)

        sigma_inv = la.inv(x[1])

        inner = 0
        for i in range(k):
            s_inv = sigma_inv[i, :, :]
            maha = np.real(u[0][i, :, :].conj().T@s_inv@v[0][i, :, :])
            inner += 2 * np.sum(1/x[2][i, :, :]) * maha
            tmp = s_inv @ u[1][i, :, :] @ s_inv @ v[1][i, :, :]
            inner += n * np.real(np.trace(tmp))
            t_inv = x[2][i, :, :]**(-1)
            inner += p * (u[2][i, :, :] * t_inv).T @ (v[2][i, :, :] * t_inv)

        assert type(man.inner(x, u, v)) == np.float64
        assert_allclose(man.inner(x, u, v), inner)

    def test_norm(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        u = man.randvec(x)
        norm = np.abs(rnd.normal(loc=2, scale=3))
        u = u * norm

        assert type(man.norm(x, u)) == np.float64
        assert_allclose(man.norm(x, u), norm)

    def test_proj(self):
        rnd.seed(123)

        k, n, p = self._k, self._n, self._p
        man = self.man
        x = man.rand()
        u = list()
        u.append(rnd.normal(size=(k, p, 1)) + 1j*rnd.normal(size=(k, p, 1)))
        u.append(rnd.normal(size=(k, p, p)) + 1j*rnd.normal(size=(k, p, p)))
        u.append(rnd.normal(size=(k, n, 1)))
        u = man.proj(x, u)

        self.check_tangent(x, u)
        assert_allclose(u[0], man.proj(x, u)[0])
        assert_allclose(u[1], man.proj(x, u)[1])
        assert_allclose(u[2], man.proj(x, u)[2])

    def test_egrad2rgrad(self):
        rnd.seed(123)

        k, n, p = self._k, self._n, self._p
        man = self.man
        x = man.rand()
        egrad = list()
        egrad.append(rnd.normal(size=(k, p, 1)) +
                     1j*rnd.normal(size=(k, p, 1)))
        egrad.append(rnd.normal(size=(k, p, p)) +
                     1j*rnd.normal(size=(k, p, p)))
        egrad.append(rnd.normal(size=(k, n, 1)))
        grad = man.egrad2rgrad(x, egrad)
        self.check_tangent(x, grad)

    def test_retr(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()

        # retr(x, 0) = x
        u = man.zerovec(x)
        r = man.retr(x, u)
        assert_allclose(r[0], x[0])
        assert_allclose(r[1], x[1])
        assert_allclose(r[2], x[2])

        # retr(x, u) = x + u.
        u = man.randvec(x)
        u = u * 1e-6
        r = man.retr(x, u)
        assert_allclose(r[0], x[0] + u[0])
        assert_allclose(r[1], x[1] + u[1])
        assert_allclose(r[2], x[2] + u[2])

        self.check_man(r)

    def test_transp(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.randvec(x)
        t_u = man.transp(x, y, u)

        self.check_tangent(y, t_u)

    def test_div_KL(self):
        rnd.seed(123)

        k, n, p = self._k, self._n, self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_KL(x1, x1)**2
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_KL(x1, x2)**2
        assert type(d) is np.float64
        assert d > 1e-2

        # invariance
        A = rnd.normal(size=(k, p, p)) + 1j*rnd.normal(size=(k, p, p))
        mu = rnd.normal(size=(k, p, 1)) + 1j*rnd.normal(size=(k, p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@multihconj(A), x1[2]]
        x2 = [A@x2[0] + mu, A@x2[1]@multihconj(A), x2[2]]
        assert_allclose(man.div_KL(x1, x2)**2, d)

        # squared divergence = sum of squared divergences
        d = 0
        for i in range(len(x1[0])):
            p1 = [x1[0][i], x1[1][i], x1[2][i]]
            p2 = [x2[0][i], x2[1][i], x2[2][i]]
            d += man.div_KL(p1, p2)**2
        assert_allclose(man.div_KL(x1, x2)**2, d)

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0], x1[2][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[2], reps=(len(x2[0]), 1, 1))
            ]
            return man.div_KL(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1, 2])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 3
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)
        assert egrad[2].shape == (n, 1)

    def test_div_alpha(self):
        rnd.seed(123)

        k, n, p = self._k, self._n, self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_alpha(x1, x1)**2
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_alpha(x1, x2)**2
        assert type(d) is np.float64
        assert d > 1e-2

        # invariance
        A = rnd.normal(size=(k, p, p)) + 1j*rnd.normal(size=(k, p, p))
        mu = rnd.normal(size=(k, p, 1)) + 1j*rnd.normal(size=(k, p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@multihconj(A), x1[2]]
        x2 = [A@x2[0] + mu, A@x2[1]@multihconj(A), x2[2]]
        assert_allclose(man.div_alpha(x1, x2)**2, d)

        # squared divergence = sum of squared divergences
        d = list()
        for i in range(len(x1[0])):
            p1 = [x1[0][i], x1[1][i], x1[2][i]]
            p2 = [x2[0][i], x2[1][i], x2[2][i]]
            d.append(-(man.div_alpha(p1, p2)**2 - np.log(k)))
        d = np.log(k) - logsumexp(d)
        assert_allclose(man.div_alpha(x1, x2)**2, d)

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0], x1[2][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[2], reps=(len(x2[0]), 1, 1))
            ]
            return man.div_alpha(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1, 2])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 3
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)
        assert egrad[2].shape == (n, 1)

    def test_div_alpha_sym(self):
        rnd.seed(123)

        k, n, p = self._k, self._n, self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_alpha_sym(x1, x1)**2
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_alpha_sym(x1, x2)**2
        assert type(d) is np.float64
        assert d > 1e-2

        # symmetry
        d1 = man.div_alpha_sym(x1, x2)
        d2 = man.div_alpha_sym(x2, x1)
        assert_allclose(d1, d2)

        # invariance
        A = rnd.normal(size=(k, p, p)) + 1j*rnd.normal(size=(k, p, p))
        mu = rnd.normal(size=(k, p, 1)) + 1j*rnd.normal(size=(k, p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@multihconj(A), x1[2]]
        x2 = [A@x2[0] + mu, A@x2[1]@multihconj(A), x2[2]]
        assert_allclose(man.div_alpha_sym(x1, x2)**2, d)

        # squared divergence = sum of squared divergences
        d = list()
        for i in range(k):
            p1 = [x1[0][i], x1[1][i], x1[2][i]]
            p2 = [x2[0][i], x2[1][i], x2[2][i]]
            d1 = man.div_alpha(p1, p2)**2
            d2 = man.div_alpha(p2, p1)**2
            tmp = (0.5*d1) + (0.5*d2)
            d.append(-(tmp - np.log(k)))
        d = np.log(k) - logsumexp(d)
        assert_allclose(man.div_alpha_sym(x1, x2)**2, d)

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0], x1[2][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[2], reps=(len(x2[0]), 1, 1))
            ]
            return man.div_alpha_sym(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1, 2])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 3
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)
        assert egrad[2].shape == (n, 1)
