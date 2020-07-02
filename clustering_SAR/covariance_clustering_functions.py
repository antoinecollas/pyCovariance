import numpy as np
import scipy as sp
import warnings
from multiprocessing import Process, Queue
from tqdm import tqdm

from .generic_functions import *
from .matrix_operators import sqrtm, invsqrtm, logm, expm

def vech_SCM(X, args):
    """ Serve to compute feature for Covariance only classification.
        We use vech opeartion to save memory space.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number of samples used for estimation
            * args = (center_vectors) where
                ** center_vectors = boolean to center vectors before computing the SCM

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """
    center_vectors = args
    if center_vectors:
        mean = np.mean(X, axis=1)
        mean = mean[:, np.newaxis, :]
        X = X - mean
    return list(vech(SCM(np.squeeze(X))))


def vech_tylerdet(𝐗, args):
    """ Serve to compute feature for Covariance only classification but robust estimation.
        We use vech opeartion to save memory space.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation
            * args = (ϵ, iter_max) for Tyler estimator, where
                ** ϵ = tolerance for convergence
                ** iter_max = number of iterations max

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """
    ϵ, iter_max = args
    𝚺, δ, iteration = tyler_estimator_covariance_normalisedet(np.squeeze(𝐗), ϵ, iter_max)
    return list(vech(𝚺))


def Wishart_distance(vh𝚺_1, vh𝚺_2, params=None):
    """ Wishart distance as described in II.B. of:
        P. Formont, F. Pascal, G. Vasile, J. Ovarlez and L. Ferro-Famil, 
        "Statistical Classification for Heterogeneous Polarimetric SAR Images," 
        in IEEE Journal of Selected Topics in Signal Processing, 
        vol. 5, no. 3, pp. 567-576, June 2011.
        doi: 10.1109/JSTSP.2010.2101579
        ----------------------------------------------------------------------
        Inputs:
        --------
            * vh𝚺_1 = a (p,) numpy array corresponding to the vech 
                    of the covariance matrix 1
            * vh𝚺_2 = a (p,) numpy array corresponding to the vech 
                    of the covariance matrix 2
            * params = scale either 'linear' or anything else for log scale

        Outputs:
        ---------
            * d = the Wishart distance between 𝚺_1 and 𝚺_2
        """
    𝚺_1 = unvech(vh𝚺_1)
    𝚺_2 = unvech(vh𝚺_2)
    d = np.log(np.abs(np.linalg.det(𝚺_2))) - np.log(np.abs(np.linalg.det(𝚺_1))) + \
        np.trace(np.linalg.inv(𝚺_2) @ 𝚺_1)
    if params=='linear':
        np.exp(np.real(d))
    else:
        return np.real(d)


def Wishart_affinity(vh𝚺_1, vh𝚺_2, params=None):
    """ Wishart affinity distance obtained as the inverse of Wishart distance
        ----------------------------------------------------------------------
        Inputs:
        --------
            * vh𝚺_1 = a (p,) numpy array corresponding to the vech 
                    of the covariance matrix 1
            * vh𝚺_2 = a (p,) numpy array corresponding to the vech 
                    of the covariance matrix 2
            * params = scale either 'linear' or anything else for log scale

        Outputs:
        ---------
            * a = the Wishart affinity between 𝚺_1 and 𝚺_2
        """
    
    if params=='linear':
        return 1/Wishart_distance(vh𝚺_1, vh𝚺_2, params)
    else:
        return -Wishart_distance(vh𝚺_1, vh𝚺_2, params)


def covariance_arithmetic_mean(𝐗_class, mean_parameters=None):
    """ Arithmetic mean as discribed in II.B. of:
        P. Formont, F. Pascal, G. Vasile, J. Ovarlez and L. Ferro-Famil, 
        "Statistical Classification for Heterogeneous Polarimetric SAR Images," 
        in IEEE Journal of Selected Topics in Signal Processing, 
        vol. 5, no. 3, pp. 567-576, June 2011.
        doi: 10.1109/JSTSP.2010.2101579
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗_class = array of shape (p, M) corresponding to 
                        samples in class
            * mean_parameters = unused here but needed for coherent coding

        Outputs:
        ---------
            * 𝛍 = the arithmetic mean
        """

    return np.mean(𝐗_class, axis=1)


# ----------------------------------------------------------------------------
# 2) Euclidean classifier: Euclidean distance + arithmetic mean
# ----------------------------------------------------------------------------
def covariance_Euclidean_distance(vh𝚺_1, vh𝚺_2, params='fro'):
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

# ----------------------------------------------------------------------------
# 3) Riemannian covariance classifier: Riemannian distance + mean
# ----------------------------------------------------------------------------
def Riemannian_distance_covariance(𝐱_1, 𝐱_2, params=None):
    """ Riemannian distance on covariance parameters
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐱_1 = a (p*(p+1)/2,) numpy array corresponding to the stack of vech 
                    of the covariance matrix for sample 1
            * 𝐱_2 = a (p*(p+1)/2,) numpy array corresponding to the stack of vech 
                    of the covariance matrix for sample 2
            * params = unused here
        Outputs:
        ---------
            * d = the distance between samples
        """
    𝚺_1 = unvech(𝐱_1)
    𝚺_2 = unvech(𝐱_2)
    i𝚺_1_sqm = invsqrtm(𝚺_1)
    d = np.linalg.norm( logm( i𝚺_1_sqm @ 𝚺_2 @ i𝚺_1_sqm ) )**2

    return np.real(d)


def compute_J(𝐗_class, p, M, isqrtm_𝐌, enable_multi=False, queue=None):
    """ A simple function to parallelise some part of the Riemannian mean"""

    M_subset = 𝐗_class.shape[1]
    J = np.zeros((p, p), dtype=complex)
    #for index in tqdm(range(M_subset)):
    for index in range(M_subset):
        J += (1/M) * logm(isqrtm_𝐌.conj().T @ unvech(𝐗_class[:, index]) @ isqrtm_𝐌)

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


def Riemannian_mean_covariance(𝐗_class, mean_parameters=None):
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
            * 𝐗_class = array of shape (p*(p+1)/2, M) corresponding to 
                        samples in class
            * mean_parameters = (ϵ, ϵ_step, tol, iter_max, enable_multi, number_of_threads) where
                * ϵ_start controls the speed of the gradient descent at first step
                * ϵ_update is the step of in line search: ϵ = ϵ_start * ϵ_update at each descending step
                * tol is the tolerance to stop the gradient descent
                * iter_max is the maximum number of iteration
                * enable_multi is a boolean to activate parrallel computation
                * number_of_threads is the number of threas for parrallel computation

        Outputs:
        ---------
            * 𝛍 = the vech of Riemannian mean
        """
    (p, M) = 𝐗_class.shape
    p = int(np.round(.5 * (-1 + np.sqrt(1 + 8 * p)))) # Size of matrices when unvech
    (ϵ_start, ϵ_update, tol, iter_max, enable_multi, number_of_threads) = mean_parameters

    # Initialisation by doing explog geometric mean
    𝐌mean = np.zeros((p,p), dtype=complex)
    for index in range(M):
        𝐌mean += logm(unvech(𝐗_class[:,index]))
    𝐌mean = expm(𝐌mean/M)
    
    # Initialise criterions of convergence
    δ = np.finfo(np.float64).max
    criterion = np.finfo(np.float64).max
    ϵ = ϵ_start
    iteration = 0

    # Loop conditions
    while (criterion>tol) and (iteration<iter_max) and (ϵ>tol):
        iteration = iteration + 1
        print("Riemannian mean over %d samples, iteration %d"%(M,iteration))

        # Computing needed terms
        sqrtm_𝐌 = sqrtm(𝐌mean)
        isqrtm_𝐌 = invsqrtm(𝐌mean)

        # Computing sum in the formula for the update
        J = wrapper_compute_J(𝐗_class, isqrtm_𝐌, enable_multi=enable_multi, number_of_threads=number_of_threads)

        # Updating mean matrix
        𝐌mean = sqrtm_𝐌.conj().T @ expm(ϵ*J) @ sqrtm_𝐌

        # Managing iterative algorithm
        criterion = np.linalg.norm(J, ord='fro')
        h = ϵ * criterion
        if h<δ:
            ϵ = ϵ_start * ϵ_update
            δ = h
        else:
            ϵ = .5 * ϵ

    return vech(𝐌mean)


# ----------------------------------------------------------------------------
# 3) SIRV distance for covariance classifier + Mean = arithmetic mean
# ----------------------------------------------------------------------------
def compute_feature_SIRV_distance(𝐗, args):
    """ Serve to compute feature for Covariance only classification with SIRV distance as described in:
        G. Vasile, J.-P. Ovarlez, F. Pascal, and C. Tison, 
        “Coherency matrixestimation of heterogeneous clutter in high-resolution polarimetricSAR images,”
        IEEE Trans. Geosci. Remote Sens., vol. 48, no. 4, pp.1809–1826, Apr. 2010.
        We concatenate the vech of Tyler, and the values of pixel in local windows
        The feature is just the vech of Tyler but we need the values of pixels for the SIRV distance.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation
            * args = (ϵ, iter_max) for Tyler estimator, where
                ** ϵ = tolerance for convergence for Tyler
                ** iter_max = number of iterations max for Tyler

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """
    ϵ, iter_max = args
    𝚺, δ, iteration = tyler_estimator_covariance(np.squeeze(𝐗), ϵ, iter_max)
    return list( np.hstack( [vech(𝚺), np.ravel(𝐗, order='F')]) )


def SIRV_distance(𝐜, 𝐱, params):
    """ SIRV distance as defined in:
        G. Vasile, J.-P. Ovarlez, F. Pascal, and C. Tison, 
        “Coherency matrixestimation of heterogeneous clutter in high-resolution polarimetricSAR images,”
        IEEE Trans. Geosci. Remote Sens., vol. 48, no. 4, pp.1809–1826, Apr. 2010.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐜 = a (p*(p+1)/2 + N*p,) numpy array corresponding to the stack of vech 
                    of the of mean of class matrix and vector values for samples which are unused here
            * 𝐱 = a (p*(p+1)/2 + N*p,) numpy array corresponding to the stack of vech 
                    of the covariance matrix and vector values for samples
            * params = (p, N) where:
                ** p = dimension of vectors
                ** N = number of samples
        Outputs:
        ---------
            * d = the distance between samples and class mean
        """

    p, N = params
    𝐌𝓌 = unvech(𝐜[:int(p*(p+1)/2)]) # Class mean matrix
    𝐌FP = unvech(𝐱[:int(p*(p+1)/2)]) # Tyler estimate on samples
    𝐤 = 𝐱[int(p*(p+1)/2):].reshape((p,N), order='F') # Samples in order to compute the distance

    𝓓_SIRV = np.log(np.abs(np.linalg.det(𝐌𝓌))) - np.log(np.abs(np.linalg.det(𝐌FP))) + (p/N) * \
            np.sum( np.diagonal(𝐤.conj().T@np.linalg.inv(𝐌𝓌)@𝐤) / np.diagonal(𝐤.conj().T@np.linalg.inv(𝐌FP)@𝐤) )
    return np.real(𝓓_SIRV)
