from multiprocessing import Process, Queue
import numpy as np
import scipy as sp
from tqdm import tqdm
import time
import warnings

from .generic_functions import *

# ----------------------------------------------------------------------------
# 1) K-means algorithm in a general multivariate context with an arbitary 
#    distance and an arbitrary way to chose clusters center
# ----------------------------------------------------------------------------
def compute_distance_k_means(𝐗, 𝛍, distance, distance_parameters, enable_multi=False, queue=None):
    # -----------------------------------------------------------
    # Definition of function to be executed in parallel (or not)
    # -----------------------------------------------------------
    (p, N) = 𝐗.shape
    (p, K) = 𝛍.shape
    𝐝 = np.empty((N, K))  # To store distance from each class
    for n in tqdm(range(N)):  # Looping on all samples
        for k in range(K):  # Looping on al classes
            𝐝[n, k] = distance(𝛍[:, k], 𝐗[:, n], distance_parameters)
    if enable_multi:
        queue.put(𝐝)
    else:
        return 𝐝


def compute_all_distances_k_means(𝐗, 𝛍, distance, distance_parameters,
                                  enable_multi=False, number_of_threads=4):
    """ A simple function to compute all distances in parallel for K-mean
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
            * 𝛍 = an array of shape (p,N) corresponding to classes centers
            * distance = function to compute distance between two samples
                         takes three arguments:
                         ** 𝐱_1 = sample 1
                         ** 𝐱_2 = sample 2
                         ** distance_parameters = parameters for distance function
            * distance_parameters = parameters for distance function
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
        args = [(𝐗_subsets[i], 𝛍, distance, distance_parameters, True, queues[i]) for i in range(number_of_threads)]
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
        𝐝 = compute_distance_k_means(𝐗, 𝛍, distance, distance_parameters)

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
    𝛍 = np.empty((p, K)).astype(complex)
    for k in range(K):
        𝛍[:, k] = 𝐗[:, indexes[k]]
    return 𝛍


def compute_mean_k_means(𝐗_class, mean_function, mean_parameters, enable_multi=False, queue=None, jobno=None):
    𝛍 = mean_function(𝐗_class, mean_parameters)
    if enable_multi:
        # Because we want to keep teh order of the means, we have to know which index it corresponds to
        # So we return it
        queue.put([𝛍, jobno]) 
    else:
        return 𝛍

 
def wrapper_compute_all_mean_parallel(𝐗, K, 𝓒, mean_function, mean_parameters, enable_multi=False):
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
                              ** mean_parameters = parameters for mean_function
            * mean_parameters = parameters for mean_function
            * enable_multi = enable or not parallel compuation
 
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
        args = [(𝐗[:, np.where(𝓒 == i)[0]], mean_function, mean_parameters, True, queues[i], i) for i in range(number_of_effective_threads)]
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
            print("Computing mean of class %d/%d " % (k+1,K))
            indexes = np.where(𝓒 == k)[0]  # Finding all samples belonging to class
            𝐗_class = 𝐗[:, indexes]
            𝛍[:,k] = compute_mean_k_means(𝐗_class, mean_function, mean_parameters)
 
    return 𝛍


def K_means_clustering_algorithm(
    X,
    K,
    distance,
    distance_parameters,
    mean_function,
    mean_parameters,
    init='Random',
    init_parameters=None,
    eps=1e-2,
    iter_max=20,
    enable_multi=False,
    enable_multi_mean=False,
    number_of_threads=4
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
            * distance = function to compute distance between two samples
                         takes three arguments:
                         ** x_1 = sample 1
                         ** x_2 = sample 2
                         ** distance_parameters = parameters for distance function
            * distance_parameters = parameters for distance function
            * mean_function = function to compute mean
                              takes two arguments:
                              ** X_class = array of shape (p, M) corresponding to 
                                           samples in class
                              ** mean_parameters = parameters for mean_function
            * mean_parameters = parameters for mean_function
            * init = either 'Random' (a sample is randomly chosen as center) or a 
                     function to initialise class. It takes two arguments:
                     ** X = samples array
                     ** init_parameters = parameters for init function
            * init_parameters = parameters for init function
            * eps = convergence error (adapted to the distance) so that at iteration i:
                    max_{mu_k(i), mu_k(i-1) : 0<k<K} distance(mu_k(i), mu_k(i-1)) < eps
            * iter_max = number of maximum iterations of algorithm
            * enable_multi = enable or not parallel compuation
            * enable_multi_mean = enable or not parallel compuation for mean computation
            * number_of_threads = number of parallel threads (cores of machine)

        Outputs:
        ---------
            * C = an array of shape (N,) with each sample with a label in {0,..., K-1}
            * mu = an array of shape (p,K) corresponding to classes centers
            * i = number of iterations done
            * delta = covnergence criterion
    """

    (p, N) = X.shape

    # -------------------------------
    # Initialisation of center means
    # -------------------------------
    if init == 'Random':
        indexes = random_index_for_initialisation(K, N)
        mu = choose_center_from_indexes(X, indexes)
    else:
        mu = init(X, init_parameters)

    # -------------------------------
    # Managing Algorithm convergence
    # -------------------------------
    delta = np.inf  # Distance between previous and present mean
    i = 0  # Iteration
    C = np.empty(N)  # To store clustering results
    time_distances = 0
    time_means = 0
    while delta > eps and i < iter_max:
        print("K-mean algorithm iteration %d" % i)
        # -----------------------------------------
        # Computing distance and assigning classes
        # -----------------------------------------   
        print("Computing distances of %d samples to %d classes' means" % (N,K))
        tb = time.time()
        d = compute_all_distances_k_means(
            X,
            mu,
            distance,
            distance_parameters,
            enable_multi,
            number_of_threads
        )
        te = time.time()
        time_distances += te-tb
        
        for n in range(N):  # Looping on all samples to select the closest class
            index = np.argmin(d[n, :])
            C[n] = index
        
        # -----------------------------------------
        # Computing new means using assigned samples
        # -----------------------------------------
        tb = time.time()
        mu_new = wrapper_compute_all_mean_parallel(
            X,
            K,
            C,
            mean_function,
            mean_parameters,
            enable_multi=enable_multi_mean
        )
        te = time.time()
        time_means += te-tb
 
        # ---------------------------------------------
        # Managing criterion for convergence on means
        # ---------------------------------------------
        delta = - np.inf
        for k in range(K):  # Looping on all classes
            delta = max(delta, distance(mu_new[:,k], mu[:, k], distance_parameters))
        mu = mu_new
 
        # Updating iteration
        i = i + 1

    print('Total time to compute distances between samples and classes:', int(time_distances), 's.')
    print('Total time to compute new means:', int(time_means), 's.')

    return (C, mu, i + 1, delta)


# ----------------------------------------------------------------------------
# 2) Spectral algorithm in a general multivariate context with an arbitary 
#    distance
# ----------------------------------------------------------------------------
def compute_distance_one_sample(𝐗, distance, distance_parameters, enable_multi=False, queue=None):
    # -----------------------------------------------------------
    # Definition of function to be executed in parallel (or not)
    # -----------------------------------------------------------
    (p, N) = 𝐗.shape
    𝐝 = np.empty((N,))  # To store distance from each sample
    for n in range(0, N):  # Looping on all samples
        𝐝[n] = distance(𝐗[:, 0], 𝐗[:, n], distance_parameters)
    if enable_multi:
        queue.put(𝐝)
    else:
        return 𝐝


def compute_all_distance_matrix(𝐗, distance, distance_parameters,
                                enable_multi=False, number_of_threads=4, number_operations_min=100):
    """ A simple function to compute all distances in parallel for spectral clustering
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
            * distance = function to compute distance between two samples
                     takes three arguments:
                 ** 𝐱_1 = sample 1
                 ** 𝐱_2 = sample 2
                 ** distance_parameters = parameters for distance function
            * distance_parameters = parameters for distance function
            * enable_multi = enable or not parallel compuation
            * number_of_threads = number of parallel threads (cores of machine)
            * number_operations_min = number of operations done at minimum per each thread
                                (if too low, there is no interest to do parallel computing)

        Outputs:
        ---------
            * 𝓐 = a (N,N) numpy array corresponding to the distance matrix
        """

    (p, N) = 𝐗.shape
    𝓐 = np.zeros((N, N))
    for n in tqdm(range(0, N - 1)):

        if enable_multi and (N - n - 1) > number_operations_min * number_of_threads:
            𝐝_list = []  # Results container
            indexes_split = np.hstack(
                [n + 1, n + int((N - n - 1) / number_of_threads) * np.arange(1, number_of_threads), N])
            # Separate data in subsets to be treated in parallel
            𝐗_subsets = []
            for t in range(1, number_of_threads + 1):
                𝐗_subsets.append(np.hstack([𝐗[:, n].reshape((p, 1)), 𝐗[:, indexes_split[t - 1]:indexes_split[t]]]))
            queues = [Queue() for i in range(number_of_threads)]  # Serves to obtain result for each thread
            args = [(𝐗_subsets[i], distance, distance_parameters, True, queues[i]) for i in range(number_of_threads)]
            jobs = [Process(target=compute_distance_one_sample, args=a) for a in args]
            # Starting parallel computation
            for j in jobs: j.start()
            # Obtaining result for each thread
            for q in queues: 𝐝_list.append(q.get())
            # Waiting for each thread to terminate
            for j in jobs: j.join()

            # Merging results
            for i in range(1, len(𝐝_list)):
                𝐝_list[i] = 𝐝_list[i][1:]
            𝐝 = np.hstack(𝐝_list)

        else:
            𝐝 = compute_distance_one_sample(𝐗[:, n:], distance, distance_parameters)

        𝓐[n, n:] = 𝐝
        𝓐[n:, n] = 𝐝

    return 𝓐


