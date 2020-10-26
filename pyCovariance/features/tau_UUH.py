import autograd
import autograd.numpy as np
import warnings

import pymanopt
from pymanopt.manifolds import ComplexGrassmann, Product, StrictlyPositiveVectors
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

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


##########  SUBOPTIMAL ESTIMATION  ##########

def estimation_tau_UUH_SCM(X, k):
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    U = U[:, :k]
    tau = np.ones((X.shape[1],1))
    return (U, tau)

##########  DISTANCE  ##########

def distance_grass(U1, U2):
    """ Function that computes the distance between two subspaces represented by their othonormal basis U1 and U2.
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

##########   MEAN     ##########

##########  CLASSES   ##########

