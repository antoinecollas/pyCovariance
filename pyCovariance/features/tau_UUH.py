import autograd
import autograd.numpy as np
import pymanopt
from pymanopt.manifolds import ComplexGrassmann, Product, StrictlyPositiveVectors
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
import warnings

from .base import BaseClassFeatures
from .covariance_texture import distance_texture_Riemannian


##########  SUBOPTIMAL ESTIMATION  ##########

def estimation_tau_UUH_SCM(X, k):
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    U = U[:, :k]
    tau = np.ones((X.shape[1],1))
    return (U, tau)


########## ESTIMATION USING RIEMANNIAN GEOMETRY ##########

def _cost(U, tau, X):
    n = X.shape[1]
    k = U.shape[1]
    res = np.sum(np.einsum('ij,ji->i', np.conj(X).T, X))
    A = np.conj(U).T@X
    res = res - np.sum(np.einsum('ij,ji->i', (tau/(1+tau))*np.conj(A).T, A)) 
    res = res + k*np.sum(np.log(1+tau))
    res = (1/n)*np.real(res)
    return res

def _egrad(U, tau, X):
    n = X.shape[1]
    k = U.shape[1]
    tau = np.real(tau)
    grad_U = -(2/n)*X@(tau/(1+tau)*X.conj().T)@U 
    A = np.conj(U).T@X
    grad_tau = -(1/n)*np.real((np.einsum('ij,ji->i', (1/(1+tau))*np.conj(A).T, A)[:, np.newaxis]- k)*(1/(1+tau)))
    grad = (grad_U, grad_tau)
    return grad

def create_cost_egrad(backend, X):
    if backend == "Autograd":
        @pymanopt.function.Callable
        def cost(U, tau):
            return _cost(U, tau, X)

        @pymanopt.function.Callable
        def egrad(U, tau):
            return tuple(np.conj(autograd.grad(cost, argnum=[0, 1])(U, tau)))

    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(U, tau):
            return _cost(U, tau, X)

        @pymanopt.function.Callable
        def egrad(U, tau):
            return _egrad(U, tau, X)
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad


def estimation_tau_UUH_RGD(X, k, autodiff):
    """ A function that estimates parameters of a 'tau UUH' model.
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * k = dimension of the subspace
            * autodiff = use or not autodiff
        Outputs:
            * U = orthogonal basis of subspace
            * tau """

    p, n = X.shape
    
    if autodiff:
        backend = 'Autograd'
    else:
        backend = 'Callable'

    cost, egrad = create_cost_egrad(backend, X)
    manifold = Product([ComplexGrassmann(p, k), StrictlyPositiveVectors(n)])

    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)
    solver = SteepestDescent()

    parameters = solver.solve(problem)
    return parameters


########## ESTIMATION ##########

def estimation_tau_UUH(X, k, tol=0.001, iter_max=100):
    """ A function that estimates parameters of a 'tau UUH' model.
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * k = dimension of subspace
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * U = orthogonal basis of subspace
            * tau """

    # Initialisation
    (p, N) = X.shape
    delta = np.inf # Distance between two iterations
    tau = np.ones((N, 1))
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    U = U[:, :k]
    iteration = 0

    while (delta>tol) and (iteration<iter_max):
        # compute tau
        tau_new = (1/k)*np.real(np.diag(X.conj().T@U@U.conj().T@X))-1
        tau_new[tau_new<=1e-10] = 1e-10

        # compute U
        pi = X@np.diag((tau_new/(1+tau_new)).reshape(-1))@X.conj().T
        U_new, _, _ = np.linalg.svd(pi)
        U_new = U_new[:, :k]
 
        # condition for stopping
        delta = np.linalg.norm(U_new@U_new.conj().T - U@U.conj().T, 'fro') / np.linalg.norm(U@U.conj().T, 'fro')
        iteration = iteration + 1

        # updating
        tau = tau_new
        U = U_new
 
    if iteration == iter_max:
        warnings.warn('Estimation algorithm did not converge')

    return (U, tau, delta, iteration)


def compute_feature_tau_UUH(X, k, args=(0.001, 100)):
    """ Serve to compute feature for 'tau UUH' model.
        We use vech operation to save memory space on covariance.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation
            * k = dimension of subspace
            * args = (ϵ, iter_max) for iterative estimator, where
                ** eps = tolerance for convergence
                ** iter_max = number of iterations max

        Outputs:
        ---------
            * x = the feature for classification
        """
    eps, iter_max = args
    U, tau, _, _ = estimation_tau_UUH(np.squeeze(X), k=k, tol=eps, iter_max=iter_max)
    U = U.reshape(-1)
    tau = tau.reshape(-1)
    return np.hstack([U, tau])


##########  DISTANCE  ##########

def distance_Grassmann(U1, U2):
    """ Function that computes the distance between two subspaces represented by their orthonormal basis U1 and U2.
        ---------------------------------------------
        Inputs:
        --------
            * U1 = array of shape (p, k) representing an orthonormal basis (k<=p)
            * U2 = array of shape (p, k) representing an orthonormal basis (k<=p)

        Outputs:
        ---------
            * distance between span(U1) and span(U2)
        """
    return np.linalg.norm(np.arccos(np.linalg.svd(U1.conj().T@U2, full_matrices=False, compute_uv=False)))


def distance_tau_UUH(x1, x2, p, k, N):
    """ Riemannian distance of 'tau UUH' model.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * x_1 = a (p*k+N,) numpy array containing subspace basis and texture parameters
            * x_2 = a (p*k+N,) numpy array containing subspace basis and texture parameters
            * p = dimension of data
            * k = dimension of subspace
            * N = number of textures

        Outputs:
        ---------
            * d = the distance between x_1 and x_2
    """
    U1 = x_1[p*k:].reshape((p, k))
    U2 = x_2[p*k:].reshape((p, k))
    tau_1 = x_1[:p*k]
    tau_2 = x_2[:p*k]
    d = np.sqrt(distance_Grassmann(U1, U2)**2+distance_texture_Riemannian(tau_1, tau_2)**2)
    return d


##########   MEAN     ##########

def mean_tau_UUH(X_class, p, k, N, mean_parameters=[1.0, 0.95, 1e-3, 100, False, 0]):
    pass


##########  CLASSES   ##########

class tauUUH(BaseClassFeatures):
    def __init__(
        self,
        p,
        k,
        N,
        estimation_args=None,
        mean_args=None
    ):
        super().__init__()
        self.p = p
        self.k = k
        self.N = N
        self.estimation_args = estimation_args
        self.mean_args = mean_args

    def __str__(self):
        return 'tau_UUH_Riemannian'

    def estimation(self, X):
        if self.estimation_args is not None:
            return compute_feature_tau_UUH(X, self.k, self.estimation_args)
        return compute_feature_tau_UUH(X, self.k)

    def distance(self, x1, x2):
        return distance_tau_UUH(x1, x2, self.p, self.k, self.N)

    def mean(self, X):
        if self.mean_args:
            return mean_tau_UUH(X, self.p, self.k, self.N, self.mean_args)
        return mean_tau_UUH(X, self.p, self.k, self.N)
