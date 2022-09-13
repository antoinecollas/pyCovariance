import autograd
import autograd.numpy as np
from autograd.numpy import linalg as la, random as rnd
from pymanopt.manifolds.hpd import HermitianPositiveDefinite
from pymanopt.manifolds.product import _ProductTangentVector

from pyCovariance.manifolds import ComplexGaussianIG
from pyCovariance.matrix_operators import\
        logm, multihconj, multiherm, multiprod, powm
from pyCovariance.testing import assert_allclose


class TestSingleComplexGaussianIGManifold():
    def __init__(self):
        self._p = p = 15
        self._alpha = alpha = 0.33
        self.man = ComplexGaussianIG(p=p, k=1, alpha=alpha)

    def check_type_dim(self, x):
        p = self._p
        assert len(x) == 2
        assert x[0].dtype == np.complex128
        assert x[0].shape == (p, 1)
        assert x[1].dtype == np.complex128
        assert x[1].shape == (p, p)

    def check_man(self, x):
        assert type(x) == list
        self.check_type_dim(x)

        # check symmetry
        assert_allclose(x[1], multiherm(x[1]))

        # check positivity
        assert (la.eigvalsh(x[1]) > 0).all()

    def check_tangent(self, x, u):
        self.check_man(x)
        assert type(u) == _ProductTangentVector
        self.check_type_dim(u)

        # check symmetry
        assert_allclose(x[1], multiherm(x[1]))

    def test_dim(self):
        man = self.man
        p = self._p
        dim = 2*p + p*p
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
        assert_allclose(self.man.inner(x, u, u), 1)

    def test_zerovec(self):
        rnd.seed(123)

        p = self._p
        man = self.man
        x = man.rand()
        u = man.zerovec(x)

        self.check_tangent(x, u)
        assert (u[0] == np.zeros((p, 1))).all()
        assert (u[1] == np.zeros((p, p))).all()

    def test_inner(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        u = man.randvec(x)
        v = man.randvec(x)

        sigma_inv = la.inv(x[1])

        inner = 2*np.real(u[0].conj().T@sigma_inv@v[0])
        inner += np.real(np.trace(sigma_inv@u[1]@sigma_inv@v[1]))
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

        p = self._p
        man = self.man
        x = man.rand()
        u = list()
        u.append(rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1)))
        u.append(rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p)))
        u = man.proj(x, u)

        self.check_tangent(x, u)
        assert_allclose(u[0], man.proj(x, u)[0])
        assert_allclose(u[1], man.proj(x, u)[1])

    def test_egrad2rgrad(self):
        rnd.seed(123)

        p = self._p
        man = self.man
        x = man.rand()
        egrad = list()
        egrad.append(rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1)))
        egrad.append(rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p)))
        grad = man.egrad2rgrad(x, egrad)
        self.check_tangent(x, grad)
        assert_allclose(0.5*x[1]@egrad[0], grad[0])
        assert_allclose(multiherm(x[1]@egrad[1]@x[1]), grad[1])

    # def test_ehess2rhess(self):
    #     n = self.n
    #     x = self.man.rand()
    #     u = self.man.randvec(x)
    #     egrad = rnd.randn(n, n)
    #     ehess = rnd.randn(n, n)
    #     hess = self.man.ehess2rhess(x, egrad, ehess, u)
    #     hess_proj = self.man.proj(x, hess)

    #     assert_allclose(hess, hess_proj)

    # def test_retr(self):
    #     rnd.seed(123)

    #     man = self.man
    #     x = man.rand()

    #     # retr(x, 0) = x
    #     u = man.zerovec(x)
    #     r = man.retr(x, u)
    #     assert_allclose(r[0], x[0])
    #     assert_allclose(r[1], x[1])

    #     # retr(x, u) = x + u.
    #     u = man.randvec(x)
    #     u = u * 1
    #     r = man.retr(x, u)
    #     assert_allclose(r[0], x[0] + u[0])
    #     assert_allclose(r[1], x[1] + u[1])

    #     self.check_man(r)

    def test_exp(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x = man.rand()

        # exp(x, 0) = x
        u = man.zerovec(x)
        r = man.exp(x, u)
        assert_allclose(r[0], x[0])
        assert_allclose(r[1], x[1])

        # exp(x, u) \approx x + u.
        u = man.randvec(x)
        u = u * 1e-5
        r = man.exp(x, u)
        assert_allclose(r[0], x[0] + u[0])
        assert_allclose(r[1], x[1] + u[1])
        self.check_man(r)

        # exp(x, u) is on the manifold
        u = man.randvec(x)
        r = man.exp(x, u)
        self.check_man(r)

        # geodesic with inital null speed on location is the geodesic
        # of Hermitian positive definite matrices
        hpd = HermitianPositiveDefinite(p)
        u = man.randvec(x)
        u[0] = man.zerovec(x)[0]
        r = man.exp(x, u)
        r_h = hpd.exp(x[1], u[1])

        assert_allclose(r[0], x[0])
        assert_allclose(r[1], r_h)
        self.check_man(r)

    def test_transp(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.randvec(x)
        t_u = man.transp(x, y, u)

        self.check_tangent(y, t_u)

    def test_div_orth(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_orth(x1, x1)
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_orth(x1, x2)
        assert type(d) is np.float64
        assert d > 1e-2

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_orth(x1, x2), d)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_orth(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

    def test_div_orth_sym(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_orth_sym(x1, x1)
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_orth_sym(x1, x2)
        assert type(d) is np.float64
        assert d > 1e-2

        # symmetry
        d1 = man.div_orth_sym(x1, x2)
        d2 = man.div_orth_sym(x2, x1)
        assert d1 == d2
        assert d1 == np.sqrt(0.5*(man.div_orth(x1, x2)**2
                                  + man.div_orth(x2, x1)**2))

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_orth_sym(x1, x2), d)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_orth_sym(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

    def test_div_scale(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_scale(x1, x1)
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_scale(x1, x2)
        assert type(d) is np.float64
        assert d > 1e-2

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_scale(x1, x2), d)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_scale(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

        # case sigma_1 == sigma_2
        x2[1] = x1[1]
        delta_mu = x2[0] - x1[0]
        s_inv = powm(x1[1], -1)
        d = 4 * (np.arccosh(1 + 0.25*delta_mu.conj().T@s_inv@delta_mu)**2)
        d = np.sqrt(np.real(d))
        assert_allclose(d, man.div_scale(x1, x2))

        # case mu_1 == mu_2
        x1 = man.rand()
        x2 = man.rand()
        x2[0] = x1[0]

        s_inv = powm(x1[1], -1)
        c = np.real(la.det(s_inv@x2[1]))**(1/p)
        d = p * np.log(c)**2
        tmp = c*powm(x2[1], -0.5)@x1[1]@powm(x2[1], -0.5)
        d = d + la.norm(logm(tmp))**2
        d = np.sqrt(np.real(d))
        assert_allclose(d, man.div_scale(x1, x2))

    def test_div_scale_sym(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_scale_sym(x1, x1)
        assert type(d) is np.float64
        assert d >= 0
        assert d < 1e-10
        d = man.div_scale_sym(x1, x2)
        assert type(d) is np.float64
        assert d > 1e-2

        # symmetry
        d1 = man.div_scale_sym(x1, x2)
        d2 = man.div_scale_sym(x2, x1)
        assert d1 == d2
        assert d1 == np.sqrt(0.5*(man.div_scale(x1, x2)**2
                                  + man.div_scale(x2, x1)**2))

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_scale_sym(x1, x2), d)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_scale_sym(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

    def test_div_KL(self):
        rnd.seed(123)

        p = self._p

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
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_KL(x1, x2)**2, d)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_KL(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

        # case sigma_1 == sigma_2: Mahalanobis distance
        x2[1] = x1[1]
        delta_mu = x2[0] - x1[0]
        s_inv = powm(x1[1], -1)
        d = delta_mu.conj().T @ s_inv @ delta_mu
        d = np.sqrt(np.real(d))
        assert_allclose(d, man.div_KL(x1, x2))

        # case mu_1 == mu_2
        x1 = man.rand()
        x2 = man.rand()
        x2[0] = x1[0]

        d = np.real(np.trace(powm(x2[1], -1)@x1[1]))
        d = d - p
        d = d + np.log(np.real(la.det(x2[1])))
        d = d - np.log(np.real(la.det(x1[1])))
        d = np.sqrt(d)
        assert_allclose(d, man.div_KL(x1, x2))

    def test_div_alpha(self):
        rnd.seed(123)

        p = self._p
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
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_alpha(x1, x2)**2, d)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_alpha(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

        # case sigma_1 == sigma_2: Mahalanobis distance
        x2[1] = x1[1]
        delta_mu = x2[0] - x1[0]
        s_inv = powm(x1[1], -1)
        d = np.real(delta_mu.conj().T @ s_inv @ delta_mu)
        d = np.exp(-alpha*(1-alpha)*d)
        d = (1/(alpha*(1-alpha))) * (1 - d)
        assert_allclose(man.div_alpha(x1, x2)**2, d)

        # case mu_1 == mu_2
        x1 = man.rand()
        x2 = man.rand()
        x2[0] = x1[0]
        det_1 = np.real(la.det(x1[1]))
        det_2 = np.real(la.det(x2[1]))
        det_mean_cov = np.real(la.det((1-alpha)*x1[1] + alpha*x2[1]))
        d = (det_1**(1-alpha) * det_2**alpha) / det_mean_cov
        d = (1/(alpha*(1-alpha))) * (1 - d)
        assert_allclose(man.div_alpha(x1, x2)**2, d)

        # divergence between totally different distributions
        x1[0] = x1[0] * 100
        x2[0] = x2[0] * -100
        assert_allclose(man.div_alpha(x1, x2)**2, 1/(alpha*(1-alpha)))

    def test_div_alpha_real_case(self):
        rnd.seed(123)

        p = self._p
        alpha = self._alpha

        man = self.man
        x1 = man.rand()
        x1[0] = x1[0].real
        assert x1[0].dtype == np.float64
        x1[1] = x1[1].real
        assert x1[1].dtype == np.float64
        x2 = man.rand()
        x2[0] = x2[0].real
        assert x2[0].dtype == np.float64
        x2[1] = x2[1].real
        assert x2[1].dtype == np.float64

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
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_alpha_real_case(x1, x2)**2, d)

        # differentiability
        theta = x1

        def _cost(*x):
            return man.div_alpha_real_case(x, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

        # case sigma_1 == sigma_2: Mahalanobis distance
        x2[1] = x1[1]
        delta_mu = x2[0] - x1[0]
        s_inv = powm(x1[1], -1)
        d = np.real(delta_mu.conj().T @ s_inv @ delta_mu)
        d = np.exp(-0.5*alpha*(1-alpha)*d)
        d = (1/(alpha*(1-alpha))) * (1 - d)
        assert_allclose(man.div_alpha_real_case(x1, x2)**2, d)

        # case mu_1 == mu_2
        x1 = man.rand()
        x2 = man.rand()
        x2[0] = x1[0]
        det_1 = np.real(la.det(x1[1]))
        det_2 = np.real(la.det(x2[1]))
        det_mean_cov = np.real(la.det((1-alpha)*x1[1] + alpha*x2[1]))
        d = (det_1**((1-alpha)/2) * det_2**(alpha/2)) / det_mean_cov**(1/2)
        d = (1/(alpha*(1-alpha))) * (1 - d)
        assert_allclose(man.div_alpha_real_case(x1, x2)**2, d)

        # real KL is 1/2 of complex KL
        x1 = man.rand()
        x2 = man.rand()
        KL_man = ComplexGaussianIG(p=p, k=1, alpha=0)
        temp0 = KL_man.div_alpha_real_case(x1, x2)**2
        temp1 = KL_man.div_alpha(x1, x2)**2
        assert_allclose(temp0, 0.5*temp1)

        # divergence between totally different distributions
        x1[0] = x1[0] * 100
        x2[0] = x2[0] * -100
        assert_allclose(
            man.div_alpha_real_case(x1, x2)**2,
            1/(alpha*(1-alpha))
        )


class TestProductComplexGaussianIGManifold():
    def __init__(self):
        self._p = p = 15
        self._k = k = 5
        self._alpha = alpha = 0.33
        self.man = ComplexGaussianIG(p=p, k=k, alpha=alpha)

    def check_type_dim(self, x):
        p = self._p
        k = self._k
        assert len(x) == 2
        assert x[0].dtype == np.complex128
        assert x[0].shape == (k, p, 1)
        assert x[1].dtype == np.complex128
        assert x[1].shape == (k, p, p)

    def check_man(self, x):
        assert type(x) == list
        self.check_type_dim(x)

        # check symmetry
        assert_allclose(x[1], multiherm(x[1]))

        # check positivity
        assert (la.eigvalsh(x[1]) > 0).all()

    def check_tangent(self, x, u):
        self.check_man(x)
        assert type(u) == _ProductTangentVector
        self.check_type_dim(u)

        # check symmetry
        assert_allclose(x[1], multiherm(x[1]))

    def test_dim(self):
        man = self.man
        p = self._p
        k = self._k
        dim = k*2*p + k*p*p
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

        p = self._p
        k = self._k
        man = self.man
        x = man.rand()
        u = man.zerovec(x)

        self.check_tangent(x, u)
        assert (u[0] == np.zeros((k, p, 1))).all()
        assert (u[1] == np.zeros((k, p, p))).all()

    def test_inner(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        u = man.randvec(x)
        v = man.randvec(x)

        sigma_inv = la.inv(x[1])
        tmp = multiprod(np.transpose(u[0].conj(), (0, 2, 1)), sigma_inv)
        tmp = 2*multiprod(tmp, v[0])
        inner = np.real(np.sum(tmp))

        tmp0 = multiprod(sigma_inv, u[1])
        tmp1 = multiprod(sigma_inv, v[1])
        tmp = np.trace(multiprod(tmp0, tmp1), axis1=1, axis2=2)
        inner += np.real(np.sum(tmp))

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

        p = self._p
        k = self._k
        man = self.man
        x = man.rand()
        u = list()
        u.append(rnd.normal(size=(k, p, 1))
                 + 1j*rnd.normal(size=(k, p, 1)))
        u.append(rnd.normal(size=(k, p, p))
                 + 1j*rnd.normal(size=(k, p, p)))
        u = man.proj(x, u)

        self.check_tangent(x, u)
        assert_allclose(u[0], man.proj(x, u)[0])
        assert_allclose(u[1], man.proj(x, u)[1])

    def test_egrad2rgrad(self):
        rnd.seed(123)

        p = self._p
        k = self._k
        man = self.man
        x = man.rand()
        egrad = list()
        egrad.append(rnd.normal(size=(k, p, 1))
                     + 1j*rnd.normal(size=(k, p, 1)))
        egrad.append(rnd.normal(size=(k, p, p))
                     + 1j*rnd.normal(size=(k, p, p)))
        grad = man.egrad2rgrad(x, egrad)
        self.check_tangent(x, grad)
        assert_allclose(0.5*multiprod(x[1], egrad[0]), grad[0])
        tmp = multiprod(multiprod(x[1], egrad[1]), x[1])
        assert_allclose(multiherm(tmp), grad[1])

    # def test_ehess2rhess(self):
    #     n = self.n
    #     x = self.man.rand()
    #     u = self.man.randvec(x)
    #     egrad = rnd.randn(n, n)
    #     ehess = rnd.randn(n, n)
    #     hess = self.man.ehess2rhess(x, egrad, ehess, u)
    #     hess_proj = self.man.proj(x, hess)

    #     assert_allclose(hess, hess_proj)

    # def test_retr(self):
    #     rnd.seed(123)

    #     man = self.man
    #     x = man.rand()

    #     # retr(x, 0) = x
    #     u = man.zerovec(x)
    #     r = man.retr(x, u)
    #     assert_allclose(r[0], x[0])
    #     assert_allclose(r[1], x[1])

    #     # retr(x, u) = x + u.
    #     u = man.randvec(x)
    #     u = u * 1
    #     r = man.retr(x, u)
    #     assert_allclose(r[0], x[0] + u[0])
    #     assert_allclose(r[1], x[1] + u[1])

    #     self.check_man(r)

    def test_exp(self):
        rnd.seed(123)

        p = self._p
        k = self._k

        man = self.man
        x = man.rand()

        # exp(x, 0) = x
        u = man.zerovec(x)
        r = man.exp(x, u)
        assert_allclose(r[0], x[0])
        assert_allclose(r[1], x[1])

        # exp(x, u) \approx x + u.
        u = man.randvec(x)
        u = u * 1e-5
        r = man.exp(x, u)
        assert_allclose(r[0], x[0] + u[0])
        assert_allclose(r[1], x[1] + u[1])
        self.check_man(r)

        # exp(x, u) is on the manifold
        u = man.randvec(x)
        r = man.exp(x, u)
        self.check_man(r)

        # geodesic with initial null speed on location is the geodesic
        # of Hermitian positive definite matrices
        hpd = HermitianPositiveDefinite(p, k=k)
        u = man.randvec(x)
        u[0] = man.zerovec(x)[0]
        r = man.exp(x, u)
        r_h = hpd.exp(x[1], u[1])

        assert_allclose(r[0], x[0])
        assert_allclose(r[1], r_h)
        self.check_man(r)

    def test_transp(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.randvec(x)
        t_u = man.transp(x, y, u)

        self.check_tangent(y, t_u)

    def test_div_orth(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_orth(x1, x1)
        assert d >= 0
        assert d < 1e-10
        d = man.div_orth(x1, x2)
        assert d > 1e-2
        assert type(d) is np.float64

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_orth(x1, x2), d)

        # squared divergence = sum of squared divergences
        d = 0
        for i in range(len(x1[0])):
            p1 = [x1[0][i], x1[1][i]]
            p2 = [x2[0][i], x2[1][i]]
            d += man.div_orth(p1, p2)**2
        d = np.sqrt(d)
        assert_allclose(d, man.div_orth(x1, x2))

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1))
            ]
            return man.div_orth(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

    def test_div_orth_sym(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_orth_sym(x1, x1)
        assert d >= 0
        assert d < 1e-10
        d = man.div_orth_sym(x1, x2)
        assert d > 1e-2
        assert type(d) is np.float64

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_orth_sym(x1, x2), d)

        # squared divergence = sum of squared divergences
        d = 0
        for i in range(len(x1[0])):
            p1 = [x1[0][i], x1[1][i]]
            p2 = [x2[0][i], x2[1][i]]
            d += (0.5*(man.div_orth(p1, p2)**2 + man.div_orth(p2, p1)**2))
        d = np.sqrt(d)
        assert_allclose(man.div_orth_sym(x1, x2), d)

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1))
            ]
            return man.div_orth_sym(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

    def test_div_scale(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_scale(x1, x1)
        assert d >= 0
        assert d < 1e-7
        d = man.div_scale(x1, x2)
        assert d > 1e-2
        assert type(d) is np.float64

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_scale(x1, x2), d)

        # squared divergence = sum of squared divergences
        d = 0
        for i in range(len(x1[0])):
            p1 = [x1[0][i], x1[1][i]]
            p2 = [x2[0][i], x2[1][i]]
            d += man.div_scale(p1, p2)**2
        d = np.sqrt(d)
        assert_allclose(d, man.div_scale(x1, x2))

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1))
            ]
            return man.div_scale(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

    def test_div_scale_sym(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_scale_sym(x1, x1)
        assert d >= 0
        assert d < 1e-7
        d = man.div_scale_sym(x1, x2)
        assert d > 1e-2
        assert type(d) is np.float64

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_scale_sym(x1, x2), d)

        # squared divergence = sum of squared divergences
        d = 0
        for i in range(len(x1[0])):
            p1 = [x1[0][i], x1[1][i]]
            p2 = [x2[0][i], x2[1][i]]
            d += (0.5*(man.div_scale(p1, p2)**2 + man.div_scale(p2, p1)**2))
        d = np.sqrt(d)
        assert_allclose(man.div_scale_sym(x1, x2), d)

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1))
            ]
            return man.div_scale_sym(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

    def test_div_KL(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        # separability and positivity
        d = man.div_KL(x1, x1)**2
        assert d >= 0
        assert d < 1e-10
        d = man.div_KL(x1, x2)
        assert d > 1e-2
        assert type(d) is np.float64

        # invariance
        A = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_KL(x1, x2), d)

        # squared divergence = sum of squared divergences
        d = 0
        for i in range(len(x1[0])):
            p1 = [x1[0][i], x1[1][i]]
            p2 = [x2[0][i], x2[1][i]]
            d += man.div_KL(p1, p2)**2
        d = np.sqrt(d)
        assert_allclose(d, man.div_KL(x1, x2))

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1))
            ]
            return man.div_KL(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

    def test_div_alpha_real_case(self):
        rnd.seed(123)

        p = self._p

        man = self.man
        x1 = man.rand()
        x1[0] = x1[0].real
        assert x1[0].dtype == np.float64
        x1[1] = x1[1].real
        assert x1[1].dtype == np.float64
        x2 = man.rand()
        x2[0] = x2[0].real
        assert x2[0].dtype == np.float64
        x2[1] = x2[1].real
        assert x2[1].dtype == np.float64

        # separability and positivity
        d = man.div_alpha_real_case(x1, x1)**2
        assert d >= 0
        assert d < 1e-10
        d = man.div_alpha_real_case(x1, x2)
        assert d > 1e-2
        assert type(d) is np.float64

        # invariance
        A = rnd.normal(size=(p, p))
        mu = rnd.normal(size=(p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@A.conj().T]
        x2 = [A@x2[0] + mu, A@x2[1]@A.conj().T]
        assert_allclose(man.div_alpha_real_case(x1, x2), d)

        # squared divergence = sum of squared divergences
        d = 0
        for i in range(len(x1[0])):
            p1 = [x1[0][i], x1[1][i]]
            p2 = [x2[0][i], x2[1][i]]
            d += man.div_alpha_real_case(p1, p2)**2
        d = np.sqrt(d)
        assert_allclose(man.div_alpha_real_case(x1, x2), d)

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1))
            ]
            return man.div_alpha_real_case(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[1].shape == (p, p)

    def test_div_alpha_sym(self):
        rnd.seed(123)

        k, p = self._k, self._p

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
        x1 = [A@x1[0] + mu, A@x1[1]@multihconj(A)]
        x2 = [A@x2[0] + mu, A@x2[1]@multihconj(A)]
        assert_allclose(man.div_alpha_sym(x1, x2)**2, d)

        # squared divergence = sum of squared divergences
        d = list()
        for i in range(len(x1[0])):
            p1 = [x1[0][i], x1[1][i]]
            p2 = [x2[0][i], x2[1][i]]
            d1 = man.div_alpha(p1, p2)**2
            d2 = man.div_alpha(p2, p1)**2
            d.append(0.5*d1 + 0.5*d2)
        d = np.sum(d)
        assert_allclose(man.div_alpha_sym(x1, x2)**2, d)

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1)),
            ]
            return man.div_alpha_sym(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[0].dtype == np.complex128
        assert egrad[1].shape == (p, p)
        assert egrad[1].dtype == np.complex128

    def test_div_alpha_sym_real_case(self):
        rnd.seed(123)

        k, p = self._k, self._p

        man = self.man
        x1 = man.rand()
        x1[0] = x1[0].real
        x1[1] = x1[1].real
        x2 = man.rand()
        x2[0] = x2[0].real
        x2[1] = x2[1].real

        # separability and positivity
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
        A = rnd.normal(size=(k, p, p))
        mu = rnd.normal(size=(k, p, 1))
        x1 = [A@x1[0] + mu, A@x1[1]@multihconj(A)]
        x2 = [A@x2[0] + mu, A@x2[1]@multihconj(A)]
        assert_allclose(man.div_alpha_sym_real_case(x1, x2)**2, d)

        # squared divergence = sum of squared divergences
        d = list()
        for i in range(len(x1[0])):
            p1 = [x1[0][i], x1[1][i]]
            p2 = [x2[0][i], x2[1][i]]
            d1 = man.div_alpha_real_case(p1, p2)**2
            d2 = man.div_alpha_real_case(p2, p1)**2
            d.append(0.5*d1 + 0.5*d2)
        d = np.sum(d)
        assert_allclose(man.div_alpha_sym_real_case(x1, x2)**2, d)

        # differentiability, test the differentiability
        # with the gradient of the variance to compute
        # a mean
        theta = [x1[0][0], x1[1][0]]

        def _cost(*theta):
            theta_batch = [
                np.tile(theta[0], reps=(len(x2[0]), 1, 1)),
                np.tile(theta[1], reps=(len(x2[0]), 1, 1)),
            ]
            return man.div_alpha_sym_real_case(theta_batch, x2)**2

        egrad = list(autograd.grad(_cost, argnum=[0, 1])(*theta))
        assert type(egrad) is list
        assert len(egrad) == 2
        assert egrad[0].shape == (p, 1)
        assert egrad[0].dtype == np.float64
        assert egrad[1].shape == (p, p)
        assert egrad[1].dtype == np.float64
