import autograd.numpy as np
from copy import deepcopy
import pymanopt
from pymanopt.manifolds.manifold import Manifold
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent


class _FeatureArray():
    def __init__(self, *shape):
        self._array = None
        self._shape = shape

    def __str__(self):
        return self._array.__str__()

    def __empty(self):
        return self._array is None

    def __len__(self):
        a = self._array
        if self.__empty():
            return 0
        return len(self._array[0])

    @property
    def shape(self):
        if self.__empty() :
            return self.__len__()
        return tuple([self._array[i].shape for i in range(len(self._array))])

    @property
    def nb_manifolds(self):
        return len(self._shape)

    def __getitem__(self, key):
        if self.__empty():
            return list()
        a = self._array
        temp = [a[i][key] for i in range(len(a))]
        if type(key) == int:
            temp = [temp[i][np.newaxis, ...] for i in range(len(temp))]
        f_a = _FeatureArray(*[temp[i].shape[1:] for i in range(len(temp))])
        f_a.append(temp)
        return f_a

    def append(self, data):
        assert type(data) in [np.ndarray, list, _FeatureArray]

        if type(data) == np.ndarray:
            data = [data]

        if type(data) == _FeatureArray:
            data = data._array

        if self._array is None:
            self._array = [None]*len(self._shape)

        assert self.nb_manifolds == len(data)

        for i, (a, d) in enumerate(zip(self._array, data)):
            assert type(d) == np.ndarray

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
            a[i] = np.squeeze(a[i])
        if self.nb_manifolds == 1:
            a = a[0]
        return a


def _feature_estimation(method):
    def wrapper(*args, **kwargs):
        # estimation
        f = method(*args, **kwargs)

        # return a _FeatureArray
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
            * estimation = function that compute feature from np.array(p, N) where p is dimension of data and N is number of data.
            * manifold = a manifold as defined in Pymanopt.
            * args_manifold = list of arguments of the manifold. e.g size of matrices.
        """
        self._name = name
        self._estimation = _feature_estimation(estimation)
        self._M_class = manifold
        self._M = manifold(*args_manifold)
        self._M._point_layout = 1
        self._args_M = args_manifold

    def __str__(self):
        """ Name of the feature"""
        return self._name

    def estimation(self, X):
        """ Serve to compute feature.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation

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
        return d

    def mean(self, X):
        """ Compute mean of features (points on manifold self.M).
            ----------------------------------------------------------------------
            Inputs:
            --------
                * X = array of shape (feature_size, M) corresponding to samples in class
            Outputs:
            ---------
                * mean = a (feature_size) array
            """
        assert type(X) is _FeatureArray
        M = self._M_class(*(self._args_M), len(X))

        def _cost(X, theta):
            #a = _transform_theta(theta)
            d_squared = M.dist(a.export(), X.export())**2
            return d_squared

        def _grad(X, theta):
            #a = _transform_theta(theta)
            theta_batch = deepcopy(theta)
            for _ in range(len(X)-1):
                theta_batch.append(theta)
            grad = M.log(theta_batch.export(), X.export())
            if type(grad) is np.ndarray:
                grad = [grad]
            grad = [-np.mean(grad[i], axis=0) for i in range(len(grad))]
            a = _FeatureArray(*[grad[i].shape[1:] for i in range(len(grad))])
            a.append(grad)
            return a

        def _create_cost_grad(X):
            def cost(theta):
                return _cost(X, theta)

            def grad(theta):
                return _grad(X, theta)

            return cost, grad

        cost, grad = _create_cost_grad(X)

        # initialisation
        theta = X[int(np.random.randint(len(X), size=1)[0])]

        # optimization
        g = grad(theta)
        for i in range(10):
            print(g)
            # exp riemann...
            #theta -= g            
            g = grad(theta)

        return theta
