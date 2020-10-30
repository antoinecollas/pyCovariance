import autograd.numpy as np
from functools import partial
import pymanopt
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
import warnings

class BaseClassFeatures:
    def __init__(self, manifold):
        """ Serve to instantiate a BaseClassFeatures object.
        ----------------------------------------------------------------------
        Input:
        --------
            * manifold = a manifold as defined in Pymanopt.
        """
        self.M = manifold

    def __str__(self):
        raise NotImplementedError

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
        raise NotImplementedError

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
        d = self.M.dist(x1, x2)
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

        def _cost(X, theta):
            d_squared = 0
            for x in X:
                d_squared += self.M.dist(theta, x)**2
            return d_squared

        def _grad(X, theta):
            grad = np.zeros_like(theta)
            for x in X:
                grad += self.M.log(theta, x)
            grad = (1/len(X))*grad
            return grad

        def _create_cost_grad(X):
            @pymanopt.function.Callable
            def cost(theta):
                return _cost(X, theta)

            @pymanopt.function.Callable
            def grad(theta):
                return _grad(X, theta)

            return cost, grad

        cost, grad = _create_cost_grad(X)
        problem = Problem(manifold=self.M, cost=cost, egrad=grad, verbosity=0)
        solver = SteepestDescent()

        i = np.random.randint(len(X), size=1)[0]
        init = X[i]
        mean_value = solver.solve(problem, x=init)

        return mean_value


def center_vectors(X):
    """ Serve to center vectors (e.g pixels).
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number of samples used for estimation

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """
    mean = np.mean(X, axis=1)
    mean = mean[:, np.newaxis]
    X = X - mean
    return X


def center_vectors_estimation(features):
    """ Center vectors before estimating features.
    -------------------------------------
        Inputs:
            --------
            * features = class of type BaseClassFeatures (e.g CovarianceTexture) 
        Outputs:
            ---------
            * the same class as in the input except the vectors are now centered before estimating
    """
    def estimation_on_centered_vectors(X, features_estimation):
        X = center_vectors(X)
        return features_estimation(X)
    features.estimation = partial(estimation_on_centered_vectors, features_estimation=features.estimation)
    return features
