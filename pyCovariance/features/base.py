import autograd.numpy as np
from copy import deepcopy
import pymanopt.manifolds as man
from pymanopt.manifolds.product import _ProductTangentVector
import warnings


class _FeatureArray():
    def __init__(self, *shape):
        self._array = None
        self._shape = shape

    def __str__(self):
        return self._array.__str__()

    def __empty(self):
        return self._array is None

    def __len__(self):
        if self.__empty():
            return 0
        return len(self._array[0])

    @property
    def dtype(self):
        if self.__empty():
            return tuple()
        return tuple([self._array[i].dtype for i in range(len(self._array))])

    @property
    def shape(self):
        if self.__empty():
            return self.__len__()
        return tuple([self._array[i].shape for i in range(len(self._array))])

    @property
    def nb_manifolds(self):
        return len(self._shape)

    def __getitem__(self, key):
        a = self._array
        temp = [a[i][key] for i in range(len(a))]
        if type(key) == int:
            temp = [temp[i][np.newaxis, ...] for i in range(len(temp))]
        f_a = _FeatureArray(*[temp[i].shape[1:] for i in range(len(temp))])
        f_a.append(temp)
        return f_a

    def append(self, data):
        assert type(data) in [np.ndarray, list, tuple, _FeatureArray]

        if type(data) == np.ndarray:
            data = [data]

        if type(data) == _FeatureArray:
            data = data._array

        if self._array is None:
            self._array = [None]*len(self._shape)

        assert self.nb_manifolds == len(data)

        for i, (a, d) in enumerate(zip(self._array, data)):
            assert type(d) == np.ndarray
            if a is not None:
                assert d.dtype == a.dtype, 'Wrong dtype !'

            # Add batch dim.
            if d.ndim == len(self._shape[i]):
                d = d[np.newaxis, ...]

            assert d.ndim == (len(self._shape[i])+1)

            if a is None:
                self._array[i] = d
            else:
                self._array[i] = np.concatenate([a, d], axis=0)

    def export(self):
        a = deepcopy(self._array)
        for i in range(len(a)):
            if len(a[i]) == 1:
                a[i] = np.squeeze(a[i], axis=0)
        if self.nb_manifolds == 1:
            a = a[0]
        return a


def _feature_estimation(method):
    def wrapper(*args, **kwargs):
        # estimation
        f = method(*args, **kwargs)

        # return a _FeatureArray
        if type(f) not in [tuple, list, np.ndarray]:
            f = np.array(f)
        if type(f) is np.ndarray:
            f = [f]
        f_a = _FeatureArray(*[f[i].shape for i in range(len(f))])
        f_a.append(f)

        return f_a
    return wrapper


class Feature():
    def __init__(self, name, estimation, manifold, args_manifold):
        """ Serve to instantiate a Feature object.
        ----------------------------------------------------------------------
        Input:
        --------
            * name = string
            * estimation = function that compute feature from np.array(p, N)
                * N is number of data.
                * p is dimension of data
            * manifold = a manifold as defined in Pymanopt.
            * args_manifold = list of arguments of the manifold.
                * e.g size of matrices.
        """
        self._name = name
        self._estimation = _feature_estimation(estimation)
        self._M_class = manifold

        if 'weights' in args_manifold:
            self._weights = args_manifold['weights']
        elif type(manifold) in [list, tuple]:
            self._weights = tuple(np.ones(len(manifold)))

        if type(manifold) in [list, tuple]:
            self._dimensions = list()
            nb_M = len(manifold)
            for i in range(nb_M):
                temp = args_manifold['sizes'][i]
                if type(temp) not in [list, tuple]:
                    temp = [temp]
                self._dimensions.append(tuple(temp))
            temp = [manifold[i](*(self._dimensions[i])) for i in range(nb_M)]
            self._M = Product(temp, self._weights)
        else:
            temp = args_manifold['sizes']
            if type(temp) not in [list, tuple]:
                temp = [temp]
            self._dimensions = tuple(temp)
            self._M = manifold(*(self._dimensions))

        self._M._point_layout = 1
        self._eps_grad = 1e-8
        self._iter_max = 100

    def __str__(self):
        """ Name of the feature"""
        return self._name

    def estimation(self, X):
        """ Serve to compute feature.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where
                * N the number
                    of samples used for estimation
                * p is the dimension of data

        Outputs:
        ---------
            * feature = a point on manifold self.M
        """
        return self._estimation(X)

    def distance(self, x1, x2):
        """ Compute distance between two features.
            ----------------------------------------------------------------------
            Inputs:
            --------
                * x1 = point n°1 on manifold self.M
                * x2 = point n°2 on manifold self.M
            Outputs:
            ---------
                * distance = a real number
            """
        assert type(x1) is _FeatureArray
        assert type(x2) is _FeatureArray
        d = self._M.dist(x1.export(), x2.export())
        if d.ndim != 0:
            d = np.squeeze(d)
        return d

    def mean(self, X):
        """ Compute mean of features (points on manifold self.M).
            ----------------------------------------------------------------------
            Inputs:
            --------
                * X = _FeatureArray
            Outputs:
            ---------
                * mean = a (feature_size) array
            """
        assert type(X) is _FeatureArray
        dim = self._dimensions
        if type(self._M_class) in [list, tuple]:
            M_class = self._M_class
            nb_M = len(M_class)
            temp = [M_class[i](*(dim[i]), len(X)) for i in range(nb_M)]
            M = Product(temp, self._weights)
        else:
            M = self._M_class(*dim, len(X))

        def _cost(X, theta):
            d_squared = M.dist(theta.export(), X.export())**2
            return d_squared

        def _minus_grad(X, theta):
            theta_batch = deepcopy(theta)
            for _ in range(len(X)-1):
                theta_batch.append(theta)
            minus_grad = M.log(theta_batch.export(), X.export())
            if type(minus_grad) is np.ndarray:
                minus_grad = [minus_grad]
            minus_grad = [np.array(np.mean(minus_grad[i], axis=0))
                          for i in range(len(minus_grad))]
            a = _FeatureArray(*[minus_grad[i].shape[1:]
                                for i in range(len(minus_grad))])
            a.append(minus_grad)
            return a

        def _create_cost_minus_grad(X):
            def cost(theta):
                return _cost(X, theta)

            def minus_grad(theta):
                return _minus_grad(X, theta)

            return cost, minus_grad

        cost, minus_grad = _create_cost_minus_grad(X)

        # initialisation
        theta = X[int(np.random.randint(len(X), size=1)[0])]
        g = minus_grad(theta)

        _iter = 0
        while ((self._M.norm(theta.export(), g.export()) > self._eps_grad) and
               (_iter < self._iter_max)):

            temp = self._M.exp(theta.export(), g.export())
            if type(temp) not in [list, np.ndarray]:
                temp = np.array(temp)
            if type(temp) is np.ndarray:
                temp = [temp]
            for i in range(len(temp)):
                if X.dtype[i] == np.float64:
                    temp[i] = temp[i].real.astype(np.float64)
            theta = _FeatureArray(*[temp[i].shape for i in range(len(temp))])
            theta.append(temp)

            g = minus_grad(theta)

            _iter += 1

        if ((self._M.norm(theta.export(), g.export()) > self._eps_grad) and
               (_iter == self._iter_max)):
            warnings.warn('Mean computation did not converge.')

        return theta


class Product(man.Product):
    """Product manifold with linear combination of metrics."""

    def __init__(self, manifolds, weights=None):
        if weights is None:
            weights = np.ones(len(manifolds))
        self._weights = tuple(weights)
        super().__init__(manifolds)

    @property
    def typicaldist(self):
        raise NotImplementedError

    def inner(self, X, G, H):
        weights = self._weights
        return np.sum([weights[k]*man.inner(X[k], G[k], H[k])
                       for k, man in enumerate(self._manifolds)])

    def dist(self, X, Y):
        weights = self._weights
        return np.sqrt(np.sum([weights[k]*(man.dist(X[k], Y[k])**2)
                               for k, man in enumerate(self._manifolds)]))

    def egrad2rgrad(self, X, U):
        weights = self._weights
        return _ProductTangentVector(
            [(1/weights[k])*man.egrad2rgrad(X[k], U[k])
             for k, man in enumerate(self._manifolds)])

    def ehess2rhess(self, X, egrad, ehess, H):
        raise NotImplementedError

    def randvec(self, X):
        weights = self._weights
        scale = len(self._manifolds) ** (-1/2)
        return _ProductTangentVector(
            [scale * (1/weights[k]**(-1/2)) * man.randvec(X[k])
             for k, man in enumerate(self._manifolds)])
