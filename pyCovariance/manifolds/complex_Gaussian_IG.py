import autograd.numpy as np
from autograd.numpy import linalg as la

from pymanopt.manifolds import\
        ComplexEuclidean,\
        HermitianPositiveDefinite
from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds.product import _ProductTangentVector

from ..matrix_operators import\
        coshm, g_invm, invsqrtm,\
        logm, powm, sinhm, sqrtm
from ..matrix_operators import\
        multihconj, multiherm, multiprod, multitrace
from .product import Product


EPS = 1e-14
ITER_MAX = 20


class ComplexGaussianIG(Manifold):
    """Information geometry of complex Gaussian distribution.
    2 parameters: location and covariance matrix.
    """
    def __init__(self, p, k=1, alpha=0.5):
        # p: dimension of data/location
        self._p = p
        # k: optimize several distributions at the same time
        self._k = k
        # alpha: only used for the alpha-divergence
        assert (alpha >= 0) and (alpha <= 1)
        self._alpha = alpha

        if k == 1:
            name = ('Information geometry of \
                    complex Gaussian distributions \
                    ({} location vectors,\
                    {} x {} covariance matrices').format(p, p, p)
        else:
            name = ('Product manifold of \
                    information geometry of \
                    complex Gaussian distributions \
                    ({} location vectors,\
                    {} x {} covariance matrices').format(p, p, p)
        dimension_location = int(2 * k * p)
        dimension_covariance = int(k * p * p)
        dimension = dimension_location + dimension_covariance
        if k == 1:
            prod_manifold = Product([
                ComplexEuclidean(p, 1),
                HermitianPositiveDefinite(p)
            ])
        else:
            prod_manifold = Product([
                ComplexEuclidean(k, p, 1),
                HermitianPositiveDefinite(p, k)
            ])
        self._prod_manifold = prod_manifold
        self._hpd = HermitianPositiveDefinite(p, k)
        super().__init__(name, dimension, point_layout=2)

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

        # location
        c = la.cholesky(x[1])
        c_inv = la.inv(c)
        res = man[0].inner(
            x[0],
            multiprod(c_inv, u[0]),
            multiprod(c_inv, v[0])
        )

        # covariance
        res = 2*res + man[1].inner(x[1], u[1], v[1])

        if self._k == 1:
            return res.reshape(1)[0]
        return res

    def norm(self, x, u):
        man = self._prod_manifold._manifolds

        # location
        c = la.cholesky(x[1])
        c_inv = la.inv(c)
        res = man[0].norm(x[0], multiprod(c_inv, u[0]))**2

        # covariance
        res = 2*res + man[1].norm(x[1], u[1])**2

        res = np.sqrt(res)

        if self._k == 1:
            return res.reshape(1)[0]
        return res

    def proj(self, x, G):
        return self._prod_manifold.proj(x, G)

    def egrad2rgrad(self, x, u):
        man = self._prod_manifold._manifolds
        grad = list()

        # location
        grad.append(0.5*multiprod(x[1], u[0]))

        # covariance
        grad.append(man[1].egrad2rgrad(x[1], u[1]))

        return _ProductTangentVector(grad)

    # def ehess2rhess(self, x, egrad, ehess, u):

    def _exp(self, x, u):
        res = list()

        s_sqrt = sqrtm(x[1])
        s_isqrt = invsqrtm(x[1])

        # G
        B = multiprod(s_isqrt, multiprod(u[1], s_isqrt))
        a = multiprod(s_isqrt, u[0])
        G_squared = powm(B, 2) + 2*multiprod(a, multihconj(a))
        G = sqrtm(G_squared)
        G_inv = g_invm(G)
        ch_G = coshm(G/2)
        sh_G = sinhm(G/2)

        # R
        temp = ch_G - multiprod(multiprod(B, G_inv), sh_G)
        R = multihconj(la.inv(temp))

        # location
        temp = 2*multiprod(multiprod(multiprod(s_sqrt, R), sh_G), G_inv)
        location = multiprod(multiprod(temp, s_isqrt), u[0]) + x[0]
        res.append(location)

        # covariance
        cov = multiprod(multiprod(multiprod(s_sqrt, R), multihconj(R)), s_sqrt)
        cov = multiherm(cov)
        res.append(cov)

        return res

    def _retr(self, x, u):
        res = list()
        s_inv = powm(x[1], -1)

        tmp = x[0] + u[0] + 0.5 * multiprod(multiprod(u[1], s_inv), u[0])
        res.append(tmp)

        tmp = multiprod(multiprod(u[1], s_inv), u[1])
        tmp = tmp - multiprod(u[0], multihconj(u[0]))
        tmp = multiherm(x[1] + u[1] + 0.5 * tmp)
        res.append(tmp)

        return res

    def _retract_on_manifold(retr_fct):
        # In the case the eigenvalues of the covariance matrix
        # are too small, it reduces the step size
        # until the retracted covariance has big enough
        # eigenvalues.
        def f(self, x, u):
            i = 0
            r = retr_fct(self, x, u)
            d = la.eigvalsh(r[1])
            while (i < ITER_MAX) and (np.sum(d < EPS) > 0):
                u = 1/2 * u
                r = retr_fct(self, x, u)
                d = la.eigvalsh(r[1])
                i += 1
            if i == ITER_MAX:
                r = x
            return r
        return f

    exp = _retract_on_manifold(_exp)

    retr = _retract_on_manifold(_retr)

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def _div_orth_helper(self, x1, x2):
        delta_mu = x2[0] - x1[0]
        s_inv = powm(x1[1], -1)

        tmp = multiprod(multiprod(multihconj(delta_mu), s_inv), delta_mu)
        tmp = np.real(tmp.squeeze())
        res_1 = np.arccosh(1 + tmp)**2

        s_proj = x1[1] + 0.5*multiprod(delta_mu, multihconj(delta_mu))
        c = invsqrtm(x2[1])
        tmp = logm(multiprod(multiprod(c, s_proj), c))
        if tmp.ndim == 2:
            res_2 = np.real(la.norm(tmp))**2
        else:
            res_2 = np.real(la.norm(tmp, axis=(1, 2)))**2

        res = np.sqrt(res_1 + res_2)
        return res

    def div_orth(self, x1, x2):
        '''Divergence built on Theorem 4 of
        TANG et al 2018, IEEE Transactions on Signal Processing
        'INFORMATION GEOMMETRIC APPROACH TO MULTISENSOR ESTIMATION FUSION'
        '''
        res = self._div_orth_helper(x1, x2)**2
        res = np.sqrt(np.sum(res))
        return res

    def div_orth_sym(self, x1, x2):
        '''Symmetrical divergence built on Theorem 4 of
        TANG et al 2018, IEEE Transactions on Signal Processing
        'INFORMATION GEOMMETRIC APPROACH TO MULTISENSOR ESTIMATION FUSION'
        '''
        res_1 = self._div_orth_helper(x1, x2)**2
        res_2 = self._div_orth_helper(x2, x1)**2
        res = np.sqrt(np.sum(0.5*(res_1+res_2)))
        return res

    def _div_scale_helper(self, x1, x2):
        p = x1[0].shape[-2]
        delta_mu = x2[0] - x1[0]
        c = np.real(la.det(multiprod(powm(x1[1], -1), x2[1]))**(1/p))

        tmp = multiprod(multiprod(multihconj(delta_mu),
                                  powm(x1[1], -1)), delta_mu)
        tmp = np.real(tmp.squeeze())
        res_1 = 4*(np.arccosh((1/(2*np.sqrt(c)))*(c + 1 + 0.5*tmp))**2)
        res_1 = res_1 + (p - 1) * (np.log(c)**2)

        if x1[1].ndim == 3:
            c = c.reshape(-1, 1, 1)

        a = invsqrtm(x2[1])
        tmp = multiherm(c*multiprod(multiprod(a, x1[1]), a))
        tmp = logm(tmp)
        if tmp.ndim == 2:
            res_2 = np.real(la.norm(tmp))**2
        else:
            res_2 = np.real(la.norm(tmp, axis=(1, 2)))**2

        res = np.sqrt(res_1 + res_2)
        return res

    def div_scale(self, x1, x2):
        '''Divergence from CALVO and OLLER 1991, Statistics & Decisions
        'AN EXPLICIT SOLUTION OF INFORMATION GEODESIC
        EQUATIONS FOR MULTIVARIATE NORMAL MODEL'
        '''
        res = self._div_scale_helper(x1, x2)**2
        res = np.sqrt(np.sum(res))
        return res

    def div_scale_sym(self, x1, x2):
        '''Symmetrical divergence based on CALVO and OLLER 1991,
        Statistics & Decisions
        'AN EXPLICIT SOLUTION OF INFORMATION GEODESIC
        EQUATIONS FOR MULTIVARIATE NORMAL MODEL'
        '''
        res_1 = self._div_scale_helper(x1, x2)**2
        res_2 = self._div_scale_helper(x2, x1)**2
        res = np.sqrt(np.sum(0.5*(res_1+res_2)))
        return res

    def _div_KL_helper(self, x1, x2):
        p = x1[0].shape[-2]
        cov_x2_inv = powm(x2[1], -1)

        # trace
        cov_x2_inv_cov_x1 = multiprod(cov_x2_inv, x1[1])
        res = np.real(multitrace(cov_x2_inv_cov_x1))

        # Mahalanobis distance
        delta_mu = x2[0] - x1[0]
        tmp = multiprod(multiprod(multihconj(delta_mu), cov_x2_inv), delta_mu)
        tmp = np.real(tmp).reshape(-1)
        res = res + np.real(tmp)

        # constant
        res = res - p

        # log-det
        tmp = np.log(la.det(cov_x2_inv_cov_x1))
        res = res - np.real(tmp)

        return res

    def div_KL(self, x1, x2):
        '''Kullback-Leibler divergence between
        complex Gaussian distributions'''
        res = self._div_KL_helper(x1, x2)
        # add 1e-12 for numerical stability
        res = np.sqrt(np.sum(res) + 1e-12)
        return res

    def _div_alpha_integral_helper(self, x1, x2):
        alpha = self._alpha

        mean_cov = (1-alpha)*x1[1] + alpha*x2[1]
        mean_cov_inv = powm(mean_cov, -1)

        # det
        det_1 = np.real(la.det(x1[1]))
        det_2 = np.real(la.det(x2[1]))
        det_mean_cov = np.real(la.det(mean_cov))
        res = ((det_1**(1-alpha)) * (det_2**alpha)) / det_mean_cov

        # Mahalanobis distance
        delta_mu = x2[0] - x1[0]
        tmp = multiprod(multihconj(delta_mu), mean_cov_inv)
        maha = np.real(multiprod(tmp, delta_mu)).reshape(-1)

        # integral
        res = res * np.exp(-alpha*(1-alpha)*maha)

        return res

    def div_alpha(self, x1, x2):
        '''alpha divergence between
        complex Gaussian distributions'''
        alpha = self._alpha

        if alpha == 0:
            return self.div_KL(x1, x2)
        if alpha == 1:
            return self.div_KL(x2, x1)

        res = self._div_alpha_integral_helper(x1, x2)
        res = 1/(alpha*(1-alpha)) * (1-res)

        # numerical stability
        res = np.abs(res)

        res = np.sqrt(np.sum(res))

        return res

    def div_alpha_sym(self, x1, x2):
        '''symmetrized alpha divergence between
        complex Gaussian distributions'''
        dir_1 = self.div_alpha(x1, x2)**2
        dir_2 = self.div_alpha(x2, x1)**2
        res = np.sqrt((0.5*dir_1) + (0.5*dir_2))
        return res

    def div_alpha_real_case(self, x1, x2):
        '''BE CAREFUL:
        alpha divergence between
        REAL Gaussian distributions'''
        alpha = self._alpha

        if alpha == 0:
            return np.sqrt(0.5*(self.div_KL(x1, x2)**2))
        if alpha == 1:
            return np.sqrt(0.5*(self.div_KL(x2, x1)**2))

        res = self._div_alpha_integral_helper(x1, x2)
        res = np.sqrt(res)
        res = 1/(alpha*(1-alpha)) * (1-res)

        # numerical stability
        res = np.abs(res)

        res = np.sqrt(np.sum(res))

        return res

    def div_alpha_sym_real_case(self, x1, x2):
        '''BE CAREFUL:
        symmetrized alpha divergence between
        REAL Gaussian distributions'''
        dir_1 = self.div_alpha_real_case(x1, x2)**2
        dir_2 = self.div_alpha_real_case(x2, x1)**2
        res = np.sqrt((0.5*dir_1) + (0.5*dir_2))
        return res

    def dist(self, x1, x2):
        # BE CAREFUL: this is not a distance but a non symmetrical divergence.
        return self.div_orth(x1, x2)
