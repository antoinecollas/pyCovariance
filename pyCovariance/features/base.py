import autograd
import autograd.numpy as np
import autograd.numpy.random as rnd
from copy import deepcopy
from pymanopt.manifolds.product import _ProductTangentVector
from pymanopt.solvers.linesearch import LineSearchBackTracking
import warnings

from ..checking import check_callable, check_positive, check_type, check_value
from ..manifolds.product import Product


class _FeatureArray():
    def __init__(self, *shape):
        self._array = None
        self._shape = shape
        self._size_preallocation = int(1e3)
        self._len = 0
        self._tol = 1e-5

    def __str__(self):
        return self._array.__str__()

    def __empty(self):
        return len(self) == 0

    def __len__(self):
        return self._len

    @property
    def dtype(self):
        if self.__empty():
            return tuple()
        return tuple([self._array[i].dtype for i in range(len(self._array))])

    @property
    def shape(self):
        if self.__empty():
            return self.__len__()
        shape = list()
        for i in range(len(self._shape)):
            shape.append((len(self), *(self._shape[i])))
        shape = tuple(shape)
        return shape

    @property
    def nb_manifolds(self):
        return len(self._shape)

    def __getitem__(self, key):
        check_positive(len(self), 'self _FeatureArray', strictly=True)

        a = self._array
        a = [a[i][:len(self)] for i in range(len(a))]
        temp = [a[i][key] for i in range(len(a))]
        if type(key) == int:
            temp = [temp[i][np.newaxis, ...] for i in range(len(temp))]
        f_a = _FeatureArray(*[temp[i].shape[1:] for i in range(len(temp))])
        f_a.append(temp)
        return f_a

    def append(self, data):
        check_type(data, 'data', [np.ndarray, list, tuple, _FeatureArray])

        if type(data) == _FeatureArray:
            data = data.export()

        if type(data) == np.ndarray:
            data = [data]

        if self._array is None:
            self._array = [None]*len(self._shape)

        check_value(self.nb_manifolds, 'self.nb_manifolds', [len(data)])

        for i, (a, d) in enumerate(zip(self._array, data)):
            check_type(d, 'd', [np.ndarray, np.memmap])
            if a is not None:
                check_value(d.dtype, 'd.dtype', [a.dtype])

            # Add batch dim.
            if len(d.shape) == len(self._shape[i]):
                d = d[np.newaxis, ...]

            check_value(d.ndim, 'd.ndim', [len(self._shape[i])+1])

            if a is None:
                self._array[i] = d
            else:
                shape = (self._size_preallocation, *(self._shape[i]))
                while len(self) + len(d) > len(self._array[i]):
                    a = self._array[i]
                    temp = np.zeros(shape, dtype=a.dtype)
                    self._array[i] = np.concatenate([a, temp], axis=0)
                self._array[i][len(self):len(self)+len(d)] = d

        self._len += len(d)

    def vectorize(self):
        check_positive(len(self), 'len(self)', strictly=True)

        temp = list()
        for a in self._array:
            vec = None
            bs = len(a)

            # check if matrix
            if a.ndim == 3:
                # check if square
                _, p, q = a.shape
                if p == q:
                    # check if matrices are symmetric or skew-symmetric
                    a_H = np.transpose(a, axes=(0, 2, 1)).conj()
                    condition_sym = (np.abs(a - a_H) < self._tol).all()
                    condition_skew = (np.abs(a + a_H) < self._tol).all()
                    if condition_sym or condition_skew:
                        indices = np.triu_indices(p)
                        a = a[:, indices[0], indices[1]]

            vec = a.reshape((bs, -1))

            temp.append(vec)

        vec = np.concatenate(temp, axis=1)

        return vec

    def export(self):
        a = [self._array[i][:len(self)] for i in range(self.nb_manifolds)]
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
        if type(f) in [np.float64, np.complex128]:
            f = np.array([f])
        if type(f) == np.ndarray:
            f = [f]
        f_a = _FeatureArray(*[f[i].shape for i in range(len(f))])
        f_a.append(f)

        return f_a
    return wrapper


def make_feature_prototype(feature):
    class TwoStepInitFeature():
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            if 'p' in kwargs:
                raise KeyError(
                    ('p should not be specified'
                     ' in arguments when calling feature.'))
            if 'N' in kwargs:
                raise KeyError(
                    ('N should not be specified'
                     ' in arguments when calling feature.'))

        def __call__(self, p, N):
            args = self.args
            kwargs = self.kwargs
            return feature(*args, **kwargs, p=p, N=N)
    return TwoStepInitFeature


class Feature():
    def __init__(self, name, estimation,
                 manifold, args_manifold,
                 init_mean=None,
                 min_grad_norm_mean=1e-4):
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
                * e.g size of matrices or weights for product manifolds.
            * init_mean = None or a function that takes points on the manifold
            * min_grad_norm_mean = minimum gradient norm
            and that outputs a point on that manifold.
            It is used to initialize the mean computation.
        """
        self._name = name.replace('.', '_')
        self._estimation = _feature_estimation(estimation)
        self._M_class = manifold

        # extract dimensions of manifolds from args_manifold
        sizes = args_manifold['sizes']
        del args_manifold['sizes']
        self._args_manifold = args_manifold
        if type(manifold) in [list, tuple]:
            self._dimensions = list()
            nb_M = len(manifold)
            for i in range(nb_M):
                temp = sizes[i]
                if type(temp) not in [list, tuple]:
                    temp = [temp]
                self._dimensions.append(tuple(temp))
            self._dimensions = tuple(self._dimensions)
        else:
            if type(sizes) not in [list, tuple]:
                sizes = [sizes]
            self._dimensions = tuple(sizes)

        # instanciate manifolds:
        # if it is a product manifold, then it instanciates
        # the different manifolds in "_prod_M" and the product
        # manifold in "_M"
        # otherwise it instanciates the manifold in "_M"
        # BE CAREFUL: the "args_manifolds" is only used in "_M"
        self._prod_M = None
        if type(manifold) in [list, tuple]:
            nb_M = len(manifold)
            self._prod_M = [manifold[i](*(self._dimensions[i]))
                            for i in range(nb_M)]
            self._M = Product(self._prod_M)
            self._M = Product(self._prod_M, **args_manifold)
        else:
            self._M = manifold(*(self._dimensions), **args_manifold)

        # hyperparameters used for mean computations
        self._M._point_layout = 1
        self._min_grad_norm = min_grad_norm_mean
        self._min_lr = 1e-8
        self._iter_max = 1000
        if init_mean is not None:
            check_callable(init_mean, 'init_mean')
        self._init_mean = init_mean

    def __str__(self):
        """ Name of the feature """
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

    def distances(self, x1, x2):
        """ Compute the different distances of a product manifold
            between two features.
            ----------------------------------------------------------------------
            Inputs:
            --------
                * x1 = point n°1 on manifold self.M
                * x2 = point n°2 on manifold self.M
            Outputs:
            ---------
                * distances = list of distances of the different manifolds
            """
        distances = [self.distance(x1, x2)]
        if self._prod_M is not None:
            if type(x1) is _FeatureArray:
                check_value(len(x1), 'len(x1)', [1])
                x1 = x1.export()
            if type(x2) is _FeatureArray:
                check_value(len(x2), 'len(x2)', [1])
                x2 = x2.export()
            for M, p1, p2 in zip(self._prod_M, x1, x2):
                d = M.dist(p1, p2)
                if d.ndim != 0:
                    d = np.squeeze(d)
                distances.append(d)
        return distances

    def distance(self, x1, x2):
        """ Compute distance between two features.
            ----------------------------------------------------------------------
            Inputs:
            --------
                * x1 = point n°1 on manifold self.M.
                * x2 = point n°2 on manifold self.M
            Outputs:
            ---------
                * distance = a real number
            """
        if type(x1) is _FeatureArray:
            check_value(len(x1), 'len(x1)', [1])
            x1 = x1.export()
        if type(x2) is _FeatureArray:
            check_value(len(x2), 'len(x2)', [1])
            x2 = x2.export()
        d = self._M.dist(x1, x2)
        if d.ndim != 0:
            d = np.squeeze(d)
        return d

    def _get_M_product(self, k):
        dim = self._dimensions
        if type(self._M_class) in [list, tuple]:
            M_class = self._M_class
            nb_M = len(M_class)
            temp = [M_class[i](*(dim[i]), k) for i in range(nb_M)]
            M = Product(temp, **(self._args_manifold))
        else:
            M = self._M_class(*dim, k, **(self._args_manifold))
        return M

    def log(self, X, Y, vectorize=False):
        """ Compute Riemannian logarithm of Y at X.
            NB:
                * X must be a single point on the manifold.
                * Y can be several points on the manifold.

            Parameters
            ----------
            X : _FeatureArray
            Y : _FeatureArray
            vectorize: bool
                If True, log is returned as a vector, i.e in coordinates.
                For example, only upper part of Hermitian matrix is returned.

            Returns
            -------
            log : np.ndarray | _FeatureArray
                log is a np.ndarray if vectorize = True.
                It is a _FeatureArray otherwise.
            """
        check_type(X, 'X', [_FeatureArray])
        check_value(len(X), 'len(X)', [1])
        check_type(Y, 'Y', [_FeatureArray])
        check_positive(len(Y), 'len(Y)', strictly=True)

        M = self._get_M_product(len(Y))

        X_batch = deepcopy(X)
        for _ in range(len(Y)-1):
            X_batch.append(X)

        temp = M.log(X_batch.export(), Y.export())
        if type(temp) is _ProductTangentVector:
            temp = list(temp)
        if type(temp) not in [list, tuple]:
            temp = [temp]

        # we often use complex manifolds on real data;
        # mathematically, this is correct but it often
        # returns data with null imaginary parts
        # hence we discard these imaginary parts
        for i in range(len(temp)):
            if X.dtype[i] == np.float64:
                temp[i] = temp[i].real.astype(np.float64)

        if len(Y) == 1:
            shape = [temp[i].shape for i in range(len(temp))]
        else:
            shape = [temp[i].shape[1:] for i in range(len(temp))]

        # return a _FeatureArray
        log = _FeatureArray(*shape)
        log.append(temp)

        if vectorize:
            log = log.vectorize()

        return log

    def mean(self, X):
        """ Compute mean of features (points on manifold self.M).
            ----------------------------------------------------------------------
            Inputs:
            --------
                * X = _FeatureArray
            Outputs:
            ---------
                * mean = _FeatureArray
            """
        check_type(X, 'X', [_FeatureArray])

        def _create_cost_grad_fct(X):

            def _cost(theta):
                M = self._get_M_product(len(X))
                if type(theta) in [np.float64, np.complex128]:
                    theta = np.array([theta])
                if type(theta) == np.ndarray:
                    theta = [theta]
                theta_batch = [
                    np.tile(
                        theta[i],
                        reps=(len(X), *([1]*theta[i].ndim))
                    )
                    for i in range(len(theta))
                ]
                if len(theta_batch) == 1:
                    theta_batch = theta_batch[0]
                d_squared = M.dist(theta_batch, X.export())**2
                d_squared = (1/(2*len(X))) * d_squared
                return d_squared

            # compute gradient in closed form
            # if the Riemannian logarithm exists
            # otherwise compute the gradient
            # using autodiff
            try:
                self.log(X[0], X[0])

                grad_closed_form = True

                def _grad(theta):
                    if type(theta) not in [list, np.ndarray]:
                        theta = np.array(theta)
                    if type(theta) is np.ndarray:
                        theta = [theta]
                    t = _FeatureArray(
                        *[theta[i].shape
                          for i in range(len(theta))])
                    t.append(theta)
                    minus_grad = self.log(t, X).export()
                    if type(minus_grad) is np.ndarray:
                        if len(X) > 1:
                            grad = -1*np.array(np.mean(minus_grad, axis=0))
                        else:
                            grad = -1*minus_grad
                    else:
                        grad = _ProductTangentVector(
                            [-1*np.array(np.mean(minus_grad[i],
                                                 axis=0))
                             for i in range(len(minus_grad))])
                    return grad

            except NotImplementedError:
                grad_closed_form = False

                def _cost_diff(*theta):
                    return _cost(theta)

                def _grad(theta):
                    if type(theta) is np.ndarray:
                        bool_theta_array = True
                        theta = [theta]
                    else:
                        bool_theta_array = False
                    argnum = list(range(len(theta)))
                    egrad = autograd.grad(_cost_diff, argnum=argnum)(*theta)
                    egrad = list(egrad)
                    for i in range(len(egrad)):
                        egrad[i] = np.conjugate(egrad[i])
                    if bool_theta_array:
                        theta = theta[0]
                        egrad = egrad[0]
                    grad = self._M.egrad2rgrad(theta, egrad)
                    if not bool_theta_array:
                        grad = _ProductTangentVector(grad)
                    return grad

            return _cost, _grad, grad_closed_form

        if len(X) == 1:
            return X

        # initialisation
        init = self._init_mean
        if init is None:
            i = int(rnd.randint(len(X), size=1)[0])
            theta = X[i].export()
        else:
            theta = init(X.export())
        if type(theta) == _FeatureArray:
            theta = theta.export()
        _iter = 0
        lr = 1

        # create cost function / gradient / linesearch
        cost, grad, grad_closed_form = _create_cost_grad_fct(X)
        if not grad_closed_form:
            linesearch = LineSearchBackTracking()

        # compute first gradient
        g = grad(theta)
        grad_norm = float(self._M.norm(theta, g))
        grad_norm_values = [grad_norm]

        # steepest descent
        while ((grad_norm > self._min_grad_norm)
               and (lr > self._min_lr)
               and (_iter < self._iter_max)):
            d = -g
            if grad_closed_form:
                if type(g) is np.ndarray:
                    d = lr * d
                else:
                    d = [lr * d[i] for i in range(len(d))]
                theta = self._M.exp(theta, d)
            else:
                lr, theta = linesearch.search(
                    cost, self._M, theta, d, cost(theta), -grad_norm**2)

            g = grad(theta)
            grad_norm = float(self._M.norm(theta, g))
            grad_norm_values.append(grad_norm)
            _iter += 1

            if grad_closed_form and (_iter in [250, 750]):
                lr = 0.1 * lr

        # check if it has converged
        if (
            ((lr <= self._min_lr)
             or ((self._min_grad_norm > 0)
                 and (_iter == self._iter_max)))
            and (grad_norm > self._min_grad_norm)
           ):
            warnings.warn('Mean computation did not converge.')
            print('Mean computation criteria:', grad_norm)
            # import matplotlib
            # import matplotlib.pyplot as plt
            # matplotlib.use('TkAgg')
            # plt.loglog(
            #     list(range(1, len(grad_norm_values)+1)), grad_norm_values)
            # plt.show()

        if type(theta) not in [list, np.ndarray]:
            theta = np.array(theta)
        if type(theta) is np.ndarray:
            theta = [theta]

        # we often use complex manifolds on real data;
        # mathematically, this is correct but it often
        # returns data with null imaginary parts
        # hence we discard these imaginary parts
        for i in range(len(theta)):
            if X.dtype[i] == np.float64:
                theta[i] = theta[i].real.astype(np.float64)

        # return a _FeatureArray
        mean = _FeatureArray(*[theta[i].shape for i in range(len(theta))])
        mean.append(theta)

        return mean
