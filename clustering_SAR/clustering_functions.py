from multiprocessing import Process, Queue
import numpy as np
import scipy as sp
from tqdm import tqdm
import time
import warnings

from .generic_functions import *

def compute_distance_k_means(
    X,
    mu,
    distance,
    enable_multi=False,
    queue=None,
    verbose=False
):
    # -----------------------------------------------------------
    # Definition of function to be executed in parallel (or not)
    # -----------------------------------------------------------
    (p, N) = X.shape
    (p, K) = mu.shape
    d = np.empty((N, K))  # To store distance from each class
    iterator = tqdm(range(N)) if verbose else range(N)
    for n in iterator:  # Looping on all samples
        for k in range(K):  # Looping on al classes
            d[n, k] = distance(mu[:, k], X[:, n])
    if enable_multi:
        queue.put(𝐝)
    else:
        return 𝐝


def compute_all_distances_k_means(
    𝐗,
    𝛍,
    distance,
    enable_multi=False,
    number_of_threads=4
):
    """ A simple function to compute all distances in parallel for K-mean
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
            * 𝛍 = an array of shape (p,N) corresponding to classes centers
            * distance = function to compute distance between two samples
                         ** 𝐱_1 = sample 1
                         ** 𝐱_2 = sample 2
            * enable_multi = enable or not parallel compuation
            * number_of_threads = number of parallel threads (cores of machine)

        Outputs:
        ---------
            * 𝐝 = a (N,K) numpy array containing all needed distances for K-mean
        """

    # -----------------------------------------------------------
    # Case: Multiprocessing is enabled
    # -----------------------------------------------------------
    if enable_multi:
        (p, N) = 𝐗.shape
        𝐝 = []  # Results container
        indexes_split = np.hstack([0, int(N / number_of_threads) * np.arange(1, number_of_threads), N])
        # Separate data in subsets to be treated in parallel
        𝐗_subsets = []
        for t in range(1, number_of_threads + 1):
            𝐗_subsets.append(𝐗[:, indexes_split[t - 1]:indexes_split[t]])
        queues = [Queue() for i in range(number_of_threads)]  # Serves to obtain result for each thread
        args = [(𝐗_subsets[i], 𝛍, distance, True, queues[i]) for i in range(number_of_threads)]
        jobs = [Process(target=compute_distance_k_means, args=a) for a in args]
        # Starting parallel computation
        for j in jobs: j.start()
        # Obtaining result for each thread
        for q in queues: 𝐝.append(q.get())
        # Waiting for each thread to terminate
        for j in jobs: j.join()

        # Merging results
        𝐝 = np.vstack(𝐝)

        # -----------------------------------------------------------
    # Case: Multiprocessing is not enabled
    # ----------------------------------------------------------- 
    else:
        𝐝 = compute_distance_k_means(𝐗, 𝛍, distance)

    return 𝐝


def random_index_for_initialisation(K, N):
    indexes = []
    for k in range(K):
        index = np.random.randint(N)
        while index in indexes:
            index = np.random.randint(N)
        indexes.append(index)
    return indexes


def choose_center_from_indexes(𝐗, indexes):
    (p, N) = 𝐗.shape
    K = len(indexes)
    𝛍 = np.empty((p, K)).astype(𝐗.dtype)
    for k in range(K):
        𝛍[:, k] = 𝐗[:, indexes[k]]
    return 𝛍


def compute_mean_k_means(
    X_class,
    mean_function,
    enable_multi=False,
    queue=None,
    jobno=None,
    verbose=False
):
    mu = mean_function(X_class)
    if enable_multi:
        # Because we want to keep teh order of the means, we have to know which index it corresponds to
        # So we return it
        queue.put([mu, jobno])
        if verbose:
            print('Mean of class', jobno+1, 'computed !')
    else:
        return mu

 
def wrapper_compute_all_mean_parallel(
    𝐗,
    K,
    𝓒,
    mean_function,
    enable_multi=False,
    verbose=False
):
    """ A simple function to compute all means in parallel for K-mean
        CAUTION: number of threads = K in this case because I did not want 
                 to bother managing the data communication
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
            * K = number of classes 
            * 𝓒 = an array of shape (N,) with each sample with a label in {0,..., K-1}
            * mean_function = function to compute mean
                              takes two arguments:
                              ** 𝐗_class = array of shape (p, M) corresponding to 
                                           samples in class
            * enable_multi = enable or not parallel compuation
            * verbose = boolean
 
        Outputs:
        ---------
            * 𝛍 = a (p,K) numpy array containing all means of classes
        """
  
    # -----------------------------------------------------------
    # Case: Multiprocessing is enabled
    # -----------------------------------------------------------
    if enable_multi:
        p = 𝐗.shape[0]
        𝛍 = np.empty((p,K), dtype=complex)
        number_of_effective_threads = K
        queues = [Queue() for i in range(number_of_effective_threads)]  # Serves to obtain result for each thread
        args = [(𝐗[:, 𝓒 == i], mean_function, True, queues[i], i) for i in range(number_of_effective_threads)]
        jobs = [Process(target=compute_mean_k_means, args=a) for a in args]
        # Starting parallel computation
        for j in jobs: j.start()
        # Obtaining result for each thread
        for q in queues: tmp=q.get(); 𝛍[:,tmp[1]] = tmp[0]
        # Waiting for each thread to terminate
        for j in jobs: j.join()
 
    # -----------------------------------------------------------
    # Case: Multiprocessing is not enabled
    # ----------------------------------------------------------- 
    else:
        p = 𝐗.shape[0]
        𝛍 = np.empty((p,K), dtype=complex)
        for k in range(K):  # Looping on all classes
            if verbose:
                print("Computing mean of class %d/%d " % (k+1,K))
            𝐗_class = 𝐗[:, 𝓒==k]
            𝛍[:,k] = compute_mean_k_means(𝐗_class, mean_function)
 
    return 𝛍

def compute_objective_function(distances):
    """ Compute the value of the objective function of K-means algorithm.
        See https://en.wikipedia.org/wiki/K-means_clustering
        ----------------------------------------------------------------------
        Inputs:
        --------
            * distances = distances between points and center of classes. np array of size (N, C) where N is the number of samples and C is the number of clusters.
        Outputs:
        ---------
            * result = value of the objective function
    """
    C = np.argmin(distances, axis=1)
    result = 0
    for k in np.unique(C):
        result += np.sum(distances[C==k, k])
    return result/distances.shape[0]

def K_means_clustering_algorithm(
    X,
    K,
    distance,
    mean_function,
    init=None,
    eps=1e-2,
    iter_max=20,
    enable_multi_distance=False,
    enable_multi_mean=False,
    number_of_threads=4,
    verbose=False
):
    """ K-means algorithm in a general multivariate context with an arbitary
        distance and an arbitray way to chose clusters center:
        Objective is to obtain a partion C = {C_0,..., C_{K-1}} of the data, 
        by computing centers mu_i and assigning samples by closest distance.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
            * K = number of classes
            * distance = function to compute distance between two samples takes two arguments:
                         ** x_1 = sample 1
                         ** x_2 = sample 2
            * mean_function = function to compute mean takes one argument:
                              ** X_class = array of shape (p, M) corresponding to 
                                           samples in class
            * init = a (N) array with one class per point (for example coming from a H-alpha decomposition). If None, centers are randomly chosen among samples.
            * iter_max = number of maximum iterations of algorithm
            * enable_multi_distance = enable or not parallel computation for distance computation
            * enable_multi_mean = enable or not parallel compuation for mean computation
            * number_of_threads = number of parallel threads (cores of machine)
            * verbose = boolean

        Outputs:
        ---------
            * C = an array of shape (N,) with each sample with a label in {0,..., K-1}
            * mu = an array of shape (p,K) corresponding to classes centers
            * i = number of iterations done
            * delta = convergence criterion
            * criterion_value = value of the objective function which is minimized (within-class variance)
    """

    (p, N) = X.shape

    # -------------------------------
    # Initialisation of center means
    # -------------------------------
    if init is None:
        indexes = random_index_for_initialisation(K, N)
        mu = choose_center_from_indexes(X, indexes)
    else:
        mu = wrapper_compute_all_mean_parallel(
            X,
            K,
            init,
            mean_function,
            enable_multi=enable_multi_mean
        )

    criterion_value = np.inf
    delta = np.inf  # Difference between previous value of criterion and new value
    i = 0  # Iteration
    C = np.empty(N)  # To store clustering results
    time_distances = 0
    time_means = 0
    
    for i in range(iter_max):
        if verbose:
            print("K-mean algorithm iteration %d" % i)

        # -----------------------------------------
        # Computing distance
        # -----------------------------------------
        if verbose:
            print("Computing distances of %d samples to %d classes' means" % (N,K))
        tb = time.time()
        d = compute_all_distances_k_means(
            X,
            mu,
            distance,
            enable_multi_distance,
            number_of_threads
        )
        te = time.time()
        time_distances += te-tb

        # -----------------------------------------
        # Assigning classes
        # -----------------------------------------   
        C = np.argmin(d, axis=1)
 
        # ---------------------------------------------
        # Managing algorithm convergence
        # ---------------------------------------------
        new_criterion_value = compute_objective_function(d)
        if verbose:
            print('############################################')
            print('K-means criterion:', round(new_criterion_value, 2))
        if criterion_value != np.inf:
            delta = np.abs(criterion_value-new_criterion_value) / criterion_value
            if delta < eps:
                if verbose:
                    print('Convergence reached:', delta)
                break
        criterion_value = new_criterion_value
        
        # -----------------------------------------
        # Computing new means using assigned samples
        # -----------------------------------------
        if verbose:
            print('############################################')
            print("Computing means of %d classes" % K)
        tb = time.time()
        mu = wrapper_compute_all_mean_parallel(
            X,
            K,
            C,
            mean_function,
            enable_multi=enable_multi_mean
        )
        te = time.time()
        time_means += te-tb
        
        if verbose:
            print()

    if verbose:
        print('Total time to compute distances between samples and classes:', int(time_distances), 's.')
        print('Total time to compute new means:', int(time_means), 's.')

    return (C, mu, i + 1, delta, criterion_value)
