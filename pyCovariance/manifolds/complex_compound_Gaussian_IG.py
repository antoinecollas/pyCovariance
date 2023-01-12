import autograd.numpy as np
from autograd.numpy import linalg as la
from autograd.scipy.special import logsumexp
from pymanopt.manifolds import\
        ComplexEuclidean,\
        HermitianPositiveDefinite,\
        SpecialHermitianPositiveDefinite
from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds.product import _ProductTangentVector

from ..matrix_operators import multihconj, multiherm, multiprod, multitrace
from ..matrix_operators import powm
from .product import Product
from .strictly_positive_vectors import\
        SpecialStrictlyPositiveVectors,\
        StrictlyPositiveVectors


EPS = 1e-18
ITER_MAX = 20


class ComplexCompoundGaussianIGConstrainedScatter(Manifold):
    """Information geometry of complex Compound Gaussian distributions
    with a unit determinant scatter matrix.
    3 parameters: location, scatter matrix, textures.
    """
    def __init__(self, p, n, k=1):
        # p: dimension of data/location
        self._p = p
        # n: number of data/ dimension of textures
        self._n = n
        # k: optimize several distributions at the same time
        # k>1: not implemented for the moment
        self._k = k
        if k > 1:
            raise NotImplementedError

        if k == 1:
            name = ('Information geometry of \
                    complex Compound Gaussian distributions \
                    ({} location vectors,\
                    {} x {} scatter matrices,\
                    {} textures vectors').format(p, p, p, n)
        else:
            name = ('Product manifold of \
                    information geometry of \
                    complex Compound Gaussian distributions \
                    ({} location vectors, \
                    {} x {} scatter matrices, \
                    {} textures vectors').format(p, p, p, n)
        dimension_location = int(2 * k * p)
        dimension_scatter = int(k * p * p - 1)
        dimension_textures = int(k * n)
        dimension = dimension_location + dimension_scatter + dimension_textures
        prod_manifold = Product([
            ComplexEuclidean(p, k),
            SpecialHermitianPositiveDefinite(p, k),
            StrictlyPositiveVectors(n, k)])
        self._prod_manifold = prod_manifold
        super().__init__(name, dimension, point_layout=3)

    def rand(self):
        return self._prod_manifold.rand()

    def randvec(self, x):
        u = self._prod_manifold.randvec(x)
        u = u * (1/self.norm(x, u))
        return u

    def zerovec(self, x):
        return self._prod_manifold.zerovec(x)

    def inner(self, x, u, v):
        man = self._prod_manifold._manifolds
        n, p = self._n, self._p

        # location
        c = la.cholesky(x[1])
        c_inv = la.inv(c)
        res = man[0].inner(
            x[0],
            multiprod(c_inv, u[0]),
            multiprod(c_inv, v[0])
        )
        res = 2 * np.sum(1/x[2], axis=0) * res

        # scatter
        res = res + n * man[1].inner(x[1], u[1], v[1])

        # textures
        res = res + p * man[2].inner(x[2], u[2], v[2])

        if self._k == 1:
            return res.reshape(1)[0]

    def norm(self, x, u):
        man = self._prod_manifold._manifolds
        k, n, p = self._k, self._n, self._p

        # location
        c = la.cholesky(x[1])
        c_inv = la.inv(c)
        res = man[0].norm(x[0], multiprod(c_inv, u[0]))**2
        res = 2 * np.sum(1/x[2], axis=0) * res

        # scatter
        res = res + n * man[1].norm(x[1], u[1])**2

        # textures
        res = res + p * man[2].norm(x[2], u[2])**2

        res = np.sqrt(res)

        if k == 1:
            return res.reshape(1)[0]

    def proj(self, x, G):
        return self._prod_manifold.proj(x, G)

    def egrad2rgrad(self, x, u):
        man = self._prod_manifold._manifolds
        n, p = self._n, self._p
        grad = list()

        # location
        grad.append((1/(2*np.sum(1/x[2])))*multiprod(x[1], u[0]))

        # scatter
        grad.append((1/n) * man[1].egrad2rgrad(x[1], u[1]))

        # textures
        grad.append((1/p) * man[2].egrad2rgrad(x[2], u[2]))

        return _ProductTangentVector(grad)

    # def ehess2rhess(self, x, egrad, ehess, u):
    #     egrad = multiherm(egrad)
    #     hess = multiprod(multiprod(x, multiherm(ehess)), x)
    #     hess += multiherm(multiprod(multiprod(u, egrad), x))
    #     return hess

    def _retr(self, x, u):
        n, p = self._n, self._p
        sigma_inv = la.inv(x[1])
        r = list()

        # location
        c = np.sum(u[2]*(x[2]**(-2))) / np.sum(1/x[2])
        c = c * np.eye(p)
        c = c + multiprod(u[1], sigma_inv)
        c = multiprod(c, u[0])
        r.append(x[0] + u[0] + (1/2) * c)

        # scatter
        c = multiprod(multiprod(u[1], sigma_inv), u[1])
        c = c - (1/n)*np.sum(1/x[2]) * multiprod(u[0], u[0].conj().T)
        sigma = x[1] + u[1] + (1/2) * c
        sigma = sigma / np.real(la.det(sigma)**(1/p))
        sigma = multiherm(sigma)
        r.append(sigma)

        # textures
        c = (u[2]**2) * (1/x[2])
        tmp = multiprod(multiprod(u[0].conj().T, sigma_inv), u[0])
        tmp = np.real(tmp[0][0])
        c = c - (1/p) * tmp * np.ones((n, 1))
        t = x[2] + u[2] + (1/2) * c
        r.append(t)

        return r

    def retr(self, x, u):
        # The retraction self._retr does not guarantee
        # to stay in the manifold. A good heuristic
        # is to control that the eigenvalues of the scatter matrix
        # and the textures are above a threshold.
        # In the case they are not, the heuristic
        # reduces the step size until the retracted point
        # is above a threshold.
        r = self._retr(x, u)
        d, _ = la.eigh(r[1])
        _iter = 0
        while _iter < ITER_MAX and np.sum(d < EPS) + np.sum(r[2] < EPS) > 0:
            _iter += 1
            u = 1/2 * u
            r = self._retr(x, u)
            d, _ = la.eigh(r[1])
        if _iter == ITER_MAX:
            return x
        return r

    def transp(self, x1, x2, d):
        return self.proj(x2, d)


class ComplexCompoundGaussianIGConstrainedTexture(Manifold):
    """Information geometry of complex Compound Gaussian distributions
    with textures with a unitary product.
    3 parameters: location, scatter matrix, textures.
    """
    def __init__(self, p, n, k=1, alpha=0.5):
        # p: dimension of data/location
        self._p = p
        # n: number of data/ dimension of textures
        self._n = n
        # k: optimize several distributions at the same time
        self._k = k
        # alpha: only used for the alpha-divergence
        assert (alpha >= 0) and (alpha <= 1)
        self._alpha = alpha

        if k == 1:
            name = ('Information geometry of \
                    complex Compound Gaussian distributions \
                    ({} location vectors,\
                    {} x {} scatter matrices,\
                    {} textures vectors').format(p, p, p, n)
        else:
            name = ('Product manifold of \
                    information geometry of \
                    complex Compound Gaussian distributions \
                    ({} location vectors, \
                    {} x {} scatter matrices, \
                    {} textures vectors').format(p, p, p, n)
        dimension_location = int(2 * k * p)
        dimension_scatter = int(k * p * p)
        dimension_textures = int(k * (n - 1))
        dimension = dimension_location + dimension_scatter + dimension_textures
        if k == 1:
            euc_man = ComplexEuclidean(p, 1)
        else:
            euc_man = ComplexEuclidean(k, p, 1)
        prod_manifold = Product([
            euc_man,
            HermitianPositiveDefinite(p, k),
            SpecialStrictlyPositiveVectors(n, k)])
        self._prod_manifold = prod_manifold
        super().__init__(name, dimension, point_layout=3)

    def rand(self):
        return self._prod_manifold.rand()

    def randvec(self, x):
        u = self._prod_manifold.randvec(x)
        u = (1 / self.norm(x, u)) * u
        return u

    def zerovec(self, x):
        return self._prod_manifold.zerovec(x)

    def inner(self, x, u, v):
        man = self._prod_manifold._manifolds
        n, p = self._n, self._p

        # location
        c = la.cholesky(x[1])
        c_inv = la.inv(c)
        tmp = np.sum(1/x[2], axis=-2, keepdims=True) * u[0]
        res = 2 * man[0].inner(
            x[0],
            multiprod(c_inv, tmp),
            multiprod(c_inv, v[0])
        )
        res = np.real(res)

        # scatter
        res = res + n * man[1].inner(x[1], u[1], v[1])

        # textures
        res = res + p * man[2].inner(x[2], u[2], v[2])

        if self._k == 1:
            res = res.reshape(1)[0]
        return res

    def norm(self, x, u):
        res = np.sqrt(self.inner(x, u, u))
        return res

    def proj(self, x, G):
        return self._prod_manifold.proj(x, G)

    def egrad2rgrad(self, x, u):
        n, p = self._n, self._p
        grad = list()

        # location
        scale = 1/(2*np.sum(1/x[2], axis=-2, keepdims=True))
        grad.append(scale * multiprod(x[1], u[0]))

        # scatter
        grad.append((1/n) * multiprod(multiprod(x[1], u[1]), x[1]))

        # textures
        grad.append((1/p) * (x[2]**2) * u[2])

        grad = _ProductTangentVector(grad)
        grad = self.proj(x, grad)

        return grad

    def _retr(self, x, u):
        n, p = self._n, self._p
        sigma_inv = la.inv(x[1])
        r = list()

        # location
        c = np.sum(u[2]*(x[2]**(-2)), axis=-2, keepdims=True)
        c = c / np.sum(1/x[2], axis=-2, keepdims=True)
        c = c * np.eye(p)
        c = c + multiprod(u[1], sigma_inv)
        c = multiprod(c, u[0])
        r.append(x[0] + u[0] + (1/2) * c)

        # scatter
        c = multiprod(multiprod(u[1], sigma_inv), u[1])
        scale = (1/n) * np.sum(1/x[2], axis=-2, keepdims=True)
        c = c - (scale * multiprod(u[0], multihconj(u[0])))
        sigma = x[1] + u[1] + (1/2) * c
        sigma = multiherm(sigma)
        r.append(sigma)

        # textures
        c = (u[2]**2) / x[2]
        tmp = multiprod(multiprod(multihconj(u[0]), sigma_inv), u[0])
        tmp = np.real(tmp[0][0])
        c = c - (1/p) * tmp * np.ones_like(x[2])
        t = x[2] + u[2] + (1/2) * c
        if np.sum(t < EPS) == 0:
            t = t / np.exp(np.mean(np.log(t)))
        r.append(t)

        return r

    def _retract_on_manifold(retr_fct):
        # In the case the eigenvalues of the covariance matrix or the textures
        # are too small, it reduces the step size
        # until the retracted covariance/textures has big enough
        # eigenvalues.
        def f(self, x, u):
            i = 0
            r = retr_fct(self, x, u)
            d = la.eigvalsh(r[1])
            while ((i < ITER_MAX)
                   and ((np.sum(d < EPS) + np.sum(r[2] < EPS)) > 0
                        or np.abs(np.exp(np.sum(np.log(r[2])))-1) > 1e-12)):
                u = 1/2 * u
                r = retr_fct(self, x, u)
                d = la.eigvalsh(r[1])
                i += 1
            if i == ITER_MAX:
                r = x
            return r
        return f

    retr = _retract_on_manifold(_retr)

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def _div_KL_helper(self, x1, x2):
        n, p = x1[2].shape[-2], x1[0].shape[-2]
        cov_x2_inv = powm(x2[1], -1)

        # trace
        t = x1[2] / x2[2]
        scale = np.sum(t, axis=-2).reshape(-1)
        cov_x2_inv_cov_x1 = multiprod(cov_x2_inv, x1[1])
        res = scale * np.real(multitrace(cov_x2_inv_cov_x1))

        # Mahalanobis distance
        delta_mu = x2[0] - x1[0]
        scale = np.sum(1/x2[2], axis=-2).reshape(-1)
        tmp = multiprod(multiprod(multihconj(delta_mu), cov_x2_inv), delta_mu)
        tmp = np.real(tmp).reshape(-1)
        res = res + (scale*tmp)

        # constant
        res = res - n*p

        # log-det
        tmp = np.log(np.real(la.det(cov_x2_inv_cov_x1)))
        res = res - n*tmp

        return res

    def div_KL(self, x1, x2):
        '''Kullback-Leibler divergence between
        complex Gaussian distributions'''
        res = self._div_KL_helper(x1, x2)

        # numerical stability
        res = np.abs(res)

        res = np.sqrt(np.sum(res))

        return res

    def _div_alpha_integral_helper(self, x1, x2):
        alpha, n = self._alpha, self._n

        # Kronecker
        tmp1 = np.einsum('...ij,...kl->...ikl', x1[2], x1[1])
        tmp2 = np.einsum('...ij,...kl->...ikl', x2[2], x2[1])

        # mean cov
        mean_cov = (1-alpha)*tmp1 + alpha*tmp2

        # det
        log_det_1 = n*(1-alpha)*np.log(np.real(la.det(x1[1])))
        log_det_2 = n*alpha*np.log(np.real(la.det(x2[1])))
        log_det_mean_cov = np.log(np.real(la.det(mean_cov)))
        log_det_mean_cov = np.sum(log_det_mean_cov, axis=-1)
        sum_log_det = log_det_1 + log_det_2 - log_det_mean_cov

        # Mahalanobis distance
        delta_mu = x2[0] - x1[0]
        mean_cov_inv = la.inv(mean_cov)
        tmp = np.sum(mean_cov_inv, axis=-3)
        tmp = multiprod(multihconj(delta_mu), tmp)
        maha = np.real(multiprod(tmp, delta_mu)).reshape(-1)
        scaled_maha = alpha*(1-alpha)*maha

        # integral
        res = sum_log_det - scaled_maha

        return res

    def div_alpha(self, x1, x2):
        '''alpha divergence between
        complex compound Gaussian distributions'''
        k = self._k
        alpha = self._alpha

        if alpha == 0:
            return self.div_KL(x1, x2)
        if alpha == 1:
            return self.div_KL(x2, x1)

        res = self._div_alpha_integral_helper(x1, x2)
        # add log(k) to have a positive value
        res = np.log(k) - logsumexp(res)

        # numerical stability
        res = np.abs(res)

        res = np.sqrt(res)

        return res

    def div_alpha_sym(self, x1, x2):
        '''symmetrized alpha divergence between
        complex compound Gaussian distributions'''
        dir_1 = self.div_alpha(x1, x2)**2
        dir_2 = self.div_alpha(x2, x1)**2
        res = np.sqrt((0.5*dir_1) + (0.5*dir_2))
        return res

    def div_alpha_real_case(self, x1, x2):
        '''BE CAREFUL:
        alpha divergence between
        REAL compound Gaussian distributions'''
        k = self._k
        alpha = self._alpha

        if alpha == 0:
            return np.sqrt(0.5*(self.div_KL(x1, x2)**2))
        if alpha == 1:
            return np.sqrt(0.5*(self.div_KL(x2, x1)**2))

        res = 0.5*self._div_alpha_integral_helper(x1, x2)
        # add log(k) to have a positive value
        res = np.log(k) - logsumexp(res)

        # numerical stability
        res = np.abs(res)

        res = np.sqrt(res)

        return res

    def div_alpha_sym_real_case(self, x1, x2):
        '''BE CAREFUL:
        symmetrized alpha divergence between
        REAL compound Gaussian distributions'''
        dir_1 = self.div_alpha_real_case(x1, x2)**2
        dir_2 = self.div_alpha_real_case(x2, x1)**2
        res = np.sqrt((0.5*dir_1) + (0.5*dir_2))
        return res

    def dist(self, x1, x2):
        # BE CAREFUL: it is a distance only if alpha = 1/2.
        return self.div_alpha(x1, x2)
