import numpy as np
import warnings


def mean(X):
    """ Compute mean of vectors
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number of samples used for estimation

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """
    mean = np.mean(X, axis=1)
    return mean


def SCM(X, *args):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * X = a matrix of size p*N with each observation along column dimension
            Outputs:
                * Sigma = the estimate"""

    (p, N) = X.shape
    return (X @ X.conj().T) / N


def tyler_estimator_covariance(X, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * sigma
            * tau
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = X.shape
    delta = np.inf # Distance between two iterations
    sigma = (1/N)*X@X.conj().T
    sigma = p*sigma/np.trace(sigma)
    iteration = 0

    # Recursive algorithm
    while (delta>tol) and (iteration<iter_max):
        # compute expression of Tyler estimator
        v = np.linalg.inv(np.linalg.cholesky(sigma))@X
        tau = np.real(np.mean(v*v.conj(),axis=0))
        X_bis = X / np.sqrt(tau)
        sigma_new = (1/N) * X_bis@X_bis.conj().T

        # imposing trace constraint: Tr(sigma) = p
        sigma_new = p*sigma_new/np.trace(sigma_new)

        # condition for stopping
        delta = np.linalg.norm(sigma_new - sigma, 'fro') / np.linalg.norm(sigma, 'fro')
        iteration = iteration + 1

        # updating sigma
        sigma = sigma_new

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (sigma, tau, delta, iteration)


def tyler_estimator_covariance_normalisedet(X, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        and normalisation by determinant
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * sigma
            * tau
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = X.shape
    delta = np.inf # Distance between two iterations
    sigma = (1/N)*X@X.conj().T
    sigma = sigma/(np.linalg.det(sigma)**(1/p))
    iteration = 0

    while (delta>tol) and (iteration<iter_max):
        # compute expression of Tyler estimator
        v = np.linalg.inv(np.linalg.cholesky(sigma))@X
        tau = np.real(np.mean(v*v.conj(),axis=0))
        X_bis = X / np.sqrt(tau)
        sigma_new = (1/N) * X_bis@X_bis.conj().T

        # imposing det constraint: det(sigma) = 1
        sigma_new = sigma_new/(np.linalg.det(sigma_new)**(1/p))

        # condition for stopping
        delta = np.linalg.norm(sigma_new - sigma, 'fro') / np.linalg.norm(sigma, 'fro')
        iteration = iteration + 1

        # updating sigma
        sigma = sigma_new
    
    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (sigma, tau, delta, iteration)


def tyler_estimator_location_covariance_normalisedet(X, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for location and covariance estimation.
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * mu = estimate of location
            * sigma = estimate of covariance
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """
    
    # Initialisation
    (p,N) = X.shape
    delta = np.inf # Distance between two iterations
    tau = np.ones((1, N))
    mu = np.mean(X, axis=1).reshape((-1, 1))
    sigma = (1/N)*(X-mu)@(X-mu).conj().T
    sigma = sigma/(np.linalg.det(sigma)**(1/p))
    iteration = 0

    while (delta>tol) and (iteration<iter_max):
        # compute tau
        v = np.linalg.inv(np.linalg.cholesky(sigma))@(X-mu)
        tau_new = np.real(np.mean(v*v.conj(), axis=0))
        
        # compute mu (location)
        mu_new = (1/np.sum(1/tau)) * np.sum(X/tau, axis=1).reshape((-1, 1))

        # compute sigma
        X_bis = (X-mu) / np.sqrt(tau)
        sigma_new = (1/N) * X_bis@X_bis.conj().T

        # imposing det constraint: det(sigma_new) = 1
        sigma_new = sigma_new/(np.linalg.det(sigma_new)**(1/p))
        
        # condition for stopping
        delta = np.linalg.norm(sigma_new - sigma, 'fro') / np.linalg.norm(sigma, 'fro')
        iteration = iteration + 1

        # updating
        tau = tau_new
        mu = mu_new
        sigma = sigma_new
    
    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (mu, sigma, tau, delta, iteration)
