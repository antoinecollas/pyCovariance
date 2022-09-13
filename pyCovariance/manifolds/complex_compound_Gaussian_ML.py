import autograd.numpy as np
from autograd.numpy import linalg as la
from pymanopt.manifolds import\
        ComplexEuclidean,\
        HermitianPositiveDefinite
from pymanopt.manifolds.manifold import Manifold

from .complex_Gaussian_IG import\
        ComplexGaussianIG
from .product import Product
from .strictly_positive_vectors import\
        SpecialStrictlyPositiveVectors


EPS = 1e-14
ITER_MAX = 20


class ComplexCompoundGaussianMLConstrainedTexture(Manifold):
    """ Geometry to perform machine learning on parameters estimated from
    complex Compound Gaussian distributions with unitary product textures.
    3 parameters: location, scatter matrix, textures.
    The Riemannian metric is a linear combination of
    - the FIM of the Gaussian distribution on the (location, scatter)
    - the FIM of the Gaussian distribution on the textures.
    """
    def __init__(self, p, n, k=1, weights=(1, 1)):
        # p: dimension of data/location
        self._p = p
        # n: number of data/ dimension of textures
        self._n = n
        # k: optimize several distributions at the same time
        self._k = k
        self._weights = weights

        if k == 1:
            name = ('Geometry to perform ML on \
                    complex Compound Gaussian distributions \
                    ({} location vectors,\
                    {} x {} scatter matrices,\
                    {} textures vectors').format(p, p, p, n)
        else:
            name = ('Product manifold to perform ML on \
                    complex Compound Gaussian distributions \
                    ({} location vectors, \
                    {} x {} scatter matrices, \
                    {} textures vectors').format(p, p, p, n)
        dimension_location = int(2 * k * p)
        dimension_scatter = int(k * p * p)
        dimension_textures = int(k * (n - 1))
        dimension = dimension_location + dimension_scatter + dimension_textures
        prod_manifold = Product([
            ComplexEuclidean(k, p, 1),
            HermitianPositiveDefinite(p, k),
            SpecialStrictlyPositiveVectors(n, k)])
        self._prod_manifold = prod_manifold
        self._loc_scatt_manifold = ComplexGaussianIG(p, k)
        self._textures_manifold = SpecialStrictlyPositiveVectors(n, k)
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
        w = self._weights
        res = w[0] * self._loc_scatt_manifold.inner(
            [x[0], x[1]], [u[0], u[1]], [v[0], v[1]])
        res = res + w[1] * self._textures_manifold.inner(
            x[2], u[2], v[2])
        return res

    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))

    def proj(self, x, G):
        return self._prod_manifold.proj(x, G)

    def egrad2rgrad(self, x, u):
        w = self._weights
        grad = self._loc_scatt_manifold.egrad2rgrad(
            [x[0], x[1]], [u[0], u[1]])
        grad.append(self._textures_manifold.egrad2rgrad(x[2], u[2]))
        grad[0] = (1/w[0]) * grad[0]
        grad[1] = (1/w[0]) * grad[1]
        grad[2] = (1/w[1]) * grad[2]

        return grad

    # def ehess2rhess(self, x, egrad, ehess, u):

    def _exp(self, x, u):
        r = self._loc_scatt_manifold._exp([x[0], x[1]], [u[0], u[1]])
        r.append(self._textures_manifold.exp(x[2], u[2]))
        return r

    def _retr(self, x, u):
        r = self._loc_scatt_manifold._retr([x[0], x[1]], [u[0], u[1]])
        r.append(self._textures_manifold.exp(x[2], u[2]))
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
                        or np.abs(np.prod(r[2])-1) > 1e-12)):
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

    def div_orth(self, x1, x2):
        '''Divergence built on Theorem 4 of
        TANG et al 2018, IEEE Transactions on Signal Processing
        'INFORMATION GEOMMETRIC APPROACH TO MULTISENSOR ESTIMATION FUSION'
        '''
        w = self._weights
        res = w[0] * (self._loc_scatt_manifold.div_orth(
            [x1[0], x1[1]], [x2[0], x2[1]])**2)
        res = res + w[1] * (self._textures_manifold.dist(x1[2], x2[2])**2)
        return np.sqrt(res)

    def div_orth_sym(self, x1, x2):
        '''Symmetrical divergence built on Theorem 4 of
        TANG et al 2018, IEEE Transactions on Signal Processing
        'INFORMATION GEOMMETRIC APPROACH TO MULTISENSOR ESTIMATION FUSION'
        '''
        w = self._weights
        res = w[0] * (self._loc_scatt_manifold.div_orth_sym(
            [x1[0], x1[1]], [x2[0], x2[1]])**2)
        res = res + w[1] * (self._textures_manifold.dist(x1[2], x2[2])**2)
        return np.sqrt(res)

    def div_scale(self, x1, x2):
        '''Divergence from CALVO and OLLER 1991, Statistics & Decisions
        'AN EXPLICIT SOLUTION OF INFORMATION GEODESIC
        EQUATIONS FOR MULTIVARIATE NORMAL MODEL'
        '''
        w = self._weights
        res = w[0] * (self._loc_scatt_manifold.div_scale(
            [x1[0], x1[1]], [x2[0], x2[1]])**2)
        res = res + w[1] * (self._textures_manifold.dist(x1[2], x2[2])**2)
        return np.sqrt(res)

    def div_scale_sym(self, x1, x2):
        '''Symmetrical divergence based on CALVO and OLLER 1991,
        Statistics & Decisions
        'AN EXPLICIT SOLUTION OF INFORMATION GEODESIC
        EQUATIONS FOR MULTIVARIATE NORMAL MODEL'
        '''
        w = self._weights
        res = w[0] * (self._loc_scatt_manifold.div_scale_sym(
            [x1[0], x1[1]], [x2[0], x2[1]])**2)
        res = res + w[1] * (self._textures_manifold.dist(x1[2], x2[2])**2)
        return np.sqrt(res)

    def dist(self, x1, x2):
        # BE CAREFUL: this is not a distance but a non symmetrical divergence.
        return self.div_orth(x1, x2)
