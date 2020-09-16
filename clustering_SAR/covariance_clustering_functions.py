import numpy as np
import scipy as sp
import warnings
from multiprocessing import Process, Queue

from .estimation import SCM
from .generic_functions import *
from .matrix_operators import sqrtm, invsqrtm, logm, expm

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

def vech_SCM(X):
    """ Serve to compute feature for Covariance only classification.
        We use vech opeartion to save memory space.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number of samples used for estimation

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """
    return vech(SCM(np.squeeze(X)))


# ----------------------------------------------------------------------------
# 2) Euclidean classifier: Euclidean distance + arithmetic mean
# ----------------------------------------------------------------------------
def distance_covariance_Euclidean(vh𝚺_1, vh𝚺_2, params='fro'):
    """ Euclidean distance between covariance matrices in parameters
        ----------------------------------------------------------------------
        Inputs:
        --------
            * vh𝚺_1 = a (p,) numpy array corresponding to the vech
                    of the covariance matrix 1
            * vh𝚺_2 = a (p,) numpy array corresponding to the vech
                    of the covariance matrix 2
            * params = order of the norm as described in:
            https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html

        Outputs:
        ---------
            * d = the Euclidean distance between 𝚺_1 and 𝚺_2
        """
    𝚺_1 = unvech(vh𝚺_1)
    𝚺_2 = unvech(vh𝚺_2)
    d = np.linalg.norm(𝚺_2-𝚺_1, params)
    return np.real(d)


def mean_covariance_Euclidean(X_class, mean_parameters=None):
    """ Arithmetic mean as discribed in II.B. of:
        P. Formont, F. Pascal, G. Vasile, J. Ovarlez and L. Ferro-Famil, 
        "Statistical Classification for Heterogeneous Polarimetric SAR Images," 
        in IEEE Journal of Selected Topics in Signal Processing, 
        vol. 5, no. 3, pp. 567-576, June 2011.
        doi: 10.1109/JSTSP.2010.2101579
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X_class = array of shape (p*(p+1)/2, M) corresponding to 
                        samples in class
            * mean_parameters = unused here but needed for coherent coding

        Outputs:
        ---------
            * the arithmetic mean
        """
    return np.mean(X_class, axis=1)


# ----------------------------------------------------------------------------
# 3) Riemannian covariance classifier: Riemannian distance + mean
# ----------------------------------------------------------------------------
def distance_covariance_Riemannian(x_1, x_2, params=None):
    """ Riemannian distance on covariance parameters
        ----------------------------------------------------------------------
        Inputs:
        --------
            * x_1 = a (p*(p+1)/2,) numpy array corresponding to the stack of vech
                    of the covariance matrix for sample 1
            * x_2 = a (p*(p+1)/2,) numpy array corresponding to the stack of vech
                    of the covariance matrix for sample 2
            * params = unused here
        Outputs:
        ---------
            * d = the distance between samples
        """
    sigma_1 = unvech(x_1)
    sigma_2 = unvech(x_2)
    eigvals = sp.linalg.eigh(sigma_2, sigma_1, eigvals_only=True)
    d = np.linalg.norm(np.log(eigvals))
    return np.real(d)


def compute_J(X_class, p, M, isqrtm_M, enable_multi=False, queue=None):
    """ A simple function to parallelise some part of the Riemannian mean"""

    M_subset = X_class.shape[1]
    J = np.zeros((p, p), dtype=complex)
    for index in range(M_subset):
        J += (1/M) * logm(isqrtm_M @ unvech(X_class[:, index]) @ isqrtm_M)

    if enable_multi:
        queue.put(J)
    else:
        return J

def wrapper_compute_J(𝐗_class, isqrtm_𝐌, enable_multi=False, number_of_threads=4):

    (p, M) = 𝐗_class.shape
    p = int(np.round(.5 * (-1 + np.sqrt(1 + 8 * p)))) # Size of matrices when unvech

    if enable_multi:        
        indexes_split = np.hstack([0, int(M / number_of_threads) * np.arange(1, number_of_threads), M])
        # Separate data in subsets to be treated in parallel
        𝐗_subsets = []
        for t in range(1, number_of_threads + 1):
            𝐗_subsets.append(𝐗_class[:, indexes_split[t - 1]:indexes_split[t]])
        queues = [Queue() for i in range(number_of_threads)]  # Serves to obtain result for each thread
        args = [(𝐗_subsets[i], p, M, isqrtm_𝐌, True, queues[i]) for i in range(number_of_threads)]
        jobs = [Process(target=compute_J, args=a) for a in args]

        J = np.zeros((p, p), dtype=complex)
        # Starting parallel computation
        for j in jobs: j.start()
        # Obtaining result for each thread
        for q in queues: J += q.get()
        # Waiting for each thread to terminate
        for j in jobs: j.join()

    else:
        J = compute_J(𝐗_class, p, M, isqrtm_𝐌)

    return J


def mean_covariance_Riemannian(X_class, mean_parameters=[1.0, 0.95, 1e-3, 30, False, 0]):
    """ Riemannian mean as discribed in section 3. of:
        P. Formont, J. P. Ovarlez, F. Pascal, G. Vasile and L. Ferro-Famil, 
        "On the extension of the product model in POLSAR processing for unsupervised 
        classification using information geometry of covariance matrices," 
        2011 IEEE International Geoscience and Remote Sensing Symposium, 
        Vancouver, BC, 2011, pp. 1361-1364.
        doi: 10.1109/IGARSS.2011.6049318
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X_class = array of shape (p*(p+1)/2, M) corresponding to 
                        samples in class
            * mean_parameters = (eps, eps_step, tol, iter_max, enable_multi, number_of_threads) where
                * eps_start controls the speed of the gradient descent at first step
                * eps_update is the step of in line search: eps = eps_start * eps_update at each descending step
                * tol is the tolerance to stop the gradient descent
                * iter_max is the maximum number of iteration
                * enable_multi is a boolean to activate parrallel computation
                * number_of_threads is the number of threas for parrallel computation

        Outputs:
        ---------
            * mu = the vech of Riemannian mean
        """
    (p, M) = X_class.shape
    if M == 0:
        raise ValueError('Can\'t compute mean with 0 value...')
    p = int(np.round(.5 * (-1 + np.sqrt(1 + 8 * p)))) # Size of matrices when unvech
    (eps_start, eps_update, tol, iter_max, enable_multi, number_of_threads) = mean_parameters

    # Initialisation by doing explog geometric mean
    mean = np.zeros((p,p), dtype=complex)
    for index in range(M):
        mean += logm(unvech(X_class[:,index]))
    mean = expm(mean/M)
    
    # Initialise criterions of convergence
    delta = np.inf
    criterion = np.inf
    eps = eps_start
    iteration = 0

    # Loop conditions
    while (criterion>tol) and (eps>tol) and (iteration<iter_max):
        iteration = iteration + 1
        if enable_multi:
            print("Riemannian mean over %d samples, iteration %d"%(M,iteration))

        # Computing needed terms
        sqrtm_M = sqrtm(mean)
        isqrtm_M = invsqrtm(mean)

        # Computing sum in the formula for the update
        J = wrapper_compute_J(X_class, isqrtm_M, enable_multi=enable_multi, number_of_threads=number_of_threads)

        # Updating mean matrix
        mean = sqrtm_M @ expm(eps*J) @ sqrtm_M

        # Managing iterative algorithm
        criterion = np.linalg.norm(J, ord='fro')
        h = eps * criterion
        if h<delta:
            eps = eps_start * eps_update
            delta = h
        else:
            eps = .5 * eps

    return vech(mean)

def mean_covariance_Riemannian_with_whitening(X_class, mean_parameters=[1.0, 0.95, 1e-3, 30, False, 0]):
    """ Riemannian mean with whitening of covariances by Euclidean mean.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X_class = array of shape (p*(p+1)/2, M) corresponding to 
                        samples in class
            * mean_parameters = (eps, eps_step, tol, iter_max, enable_multi, number_of_threads) where
                * eps_start controls the speed of the gradient descent at first step
                * eps_update is the step of in line search: eps = eps_start * eps_update at each descending step
                * tol is the tolerance to stop the gradient descent
                * iter_max is the maximum number of iteration
                * enable_multi is a boolean to activate parrallel computation
                * number_of_threads is the number of threas for parrallel computation

        Outputs:
        ---------
            * mu = the vech of Riemannian mean
        """
    (size_vech_cov, M) = X_class.shape
    if M == 0:
        raise ValueError('Can\'t compute mean with 0 value...')
    p = int(np.round(.5 * (-1 + np.sqrt(1 + 8 * size_vech_cov)))) # Size of matrices when unvech

    # whitening of data
    euclidean_mean = unvech(mean_covariance_Euclidean(X_class))
    isqrt_e_mean = invsqrtm(euclidean_mean)
    covs = np.zeros((size_vech_cov, M), dtype=np.complex)
    for i in range(M):
        covs[:, i] =  vech(isqrt_e_mean@unvech(X_class[:, i])@isqrt_e_mean)

    # Riemannian mean
    m = mean_covariance_Riemannian(covs, mean_parameters)
    sqrt_e_mean = sqrtm(euclidean_mean)
    m = vech(sqrt_e_mean@unvech(m)@sqrt_e_mean)

    return m
