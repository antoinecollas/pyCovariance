import autograd.numpy as np
from multiprocessing import Process, Queue
from tqdm import tqdm
import time
import warnings

from pyCovariance.features.base import Feature, _FeatureArray


def compute_pairwise_distances(
    X,
    mu,
    distance,
    queue=None,
    verbose=False
):
    N = len(X)
    K = len(mu)
    d = np.empty((N, K))  # To store distance from each class
    iterator = tqdm(range(N)) if verbose else range(N)
    for n in iterator:  # Looping on all samples
        for k in range(K):  # Looping on all classes
            d[n, k] = distance(X[n], mu[k])
    if queue is None:
        return d
    else:
        queue.put(d)


def compute_pairwise_distances_parallel(
    X,
    mu,
    distance,
    nb_threads=1,
    verbose=False
):
    """ A simple function to compute all distances in parallel for K-mean
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a list of N data points
            * mu = an list of data points corresponding to classes centers
            * distance = function to compute distance between two samples
                         ** x_1 = sample 1
                         ** x_2 = sample 2
            * nb_threads = number of parallel threads (cores of machine)

        Outputs:
        ---------
            * d = a (N, K) numpy array containing all distances
        """

    # -----------------------------------------------------------
    # Case: Multiprocessing is enabled
    # -----------------------------------------------------------
    if nb_threads > 1:
        N = len(X)
        d = list()
        indexes_split = np.hstack([0, int(N / nb_threads)
                                   * np.arange(1, nb_threads), N])
        # Separate data in subsets to be treated in parallel
        X_subsets = list()
        for t in range(1, nb_threads + 1):
            temp = list()
            for i in range(indexes_split[t-1], indexes_split[t]):
                temp.append(X[i])
            X_subsets.append(temp)
        queues = [Queue() for i in range(nb_threads)]
        args = [(X_subsets[i], mu, distance, queues[i], verbose)
                for i in range(nb_threads)]
        jobs = [Process(target=compute_pairwise_distances, args=a)
                for a in args]
        # Starting parallel computation
        for j in jobs:
            j.start()
        # Obtaining result for each thread
        for q in queues:
            d.append(q.get())
        # Waiting for each thread to terminate
        for j in jobs:
            j.join()

        # Merging results
        d = np.vstack(d)

        # -----------------------------------------------------------
    # Case: Multiprocessing is not enabled
    # -----------------------------------------------------------
    else:
        d = compute_pairwise_distances(X, mu, distance, None, verbose)

    return d


def compute_means(
    X_class,
    mean_function,
    queue=None,
    jobno=None,
    verbose=False
):
    mu = mean_function(X_class)
    if queue is None:
        return mu
    else:
        queue.put([mu, jobno])
        if verbose:
            print('Mean of class', jobno+1, 'computed !')


def compute_means_parallel(
    X,
    C,
    mean_function,
    nb_threads,
    verbose=False
):
    """ A simple function to compute all means in parallel for K-mean
        CAUTION: number of threads = K in this case
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = an np array of N data points
            * C = an array of shape (N,)
            with each sample with a label in {0,..., K-1}
            * mean_function = function to compute mean and takes as input:
                              ** X_class = a list of M data points
                              corresponding to samples in class
            * nb_threads = maximum number of threads
            * verbose = boolean

        Outputs:
        ---------
            * mu = a list containing all means of classes
        """

    K = len(np.unique(C))

    # -----------------------------------------------------------
    # Case: Multiprocessing is enabled
    # -----------------------------------------------------------
    if nb_threads > 1:
        means_to_compute = np.arange(K)
        mu = None

        while len(means_to_compute) != 0:
            nb_means = min(nb_threads, len(means_to_compute))
            classes = means_to_compute[:nb_means]
            means_to_compute = means_to_compute[nb_means:]

            mu_temp = [None for _ in classes]
            queues = [Queue() for _ in classes]
            args = list()
            for i, k in enumerate(classes):
                X_class = X[C == k]
                args.append((X_class, mean_function, queues[i], i))
            jobs = [Process(target=compute_means, args=a) for a in args]
            # Starting parallel computation
            for j in jobs:
                j.start()
            # Obtaining result for each thread
            for q in queues:
                tmp = q.get()
                mu_temp[tmp[1]] = tmp[0]
            mu2 = mu_temp[0]
            for i in range(1, len(classes)):
                mu2.append(mu_temp[i])
            mu_temp = mu2
            # Waiting for each thread to terminate
            for j in jobs:
                j.join()

            if mu is None:
                mu = mu_temp
            else:
                mu.append(mu_temp)

    # -----------------------------------------------------------
    # Case: Multiprocessing is not enabled
    # -----------------------------------------------------------
    else:
        mu = None
        for k in range(K):  # Looping on all classes
            if verbose:
                print("Computing mean of class %d/%d " % (k+1, K))
            X_class = X[C == k]
            temp = compute_means(X_class, mean_function)
            if mu is None:
                mu = temp
            else:
                mu.append(temp)

    return mu


def random_index_for_initialisation(K, N):
    indexes = list()
    for k in range(K):
        index = np.random.randint(N)
        while index in indexes:
            index = np.random.randint(N)
        indexes.append(index)
    return indexes


def compute_objective_function(distances):
    """ Compute the value of the objective function of K-means algorithm.
        See https://en.wikipedia.org/wiki/K-means_clustering
        ----------------------------------------------------------------------
        Inputs:
        --------
            * distances = distances between points and center of classes.
            np array of size (N, C)
        Outputs:
        ---------
            * result = value of the objective function
    """
    C = np.argmin(distances, axis=1)
    result = 0
    for k in np.unique(C):
        result += np.sum(distances[C == k, k]**2)
    return result


def _K_means(
    X,
    K,
    distance,
    mean_function,
    init=None,
    eps=1e-2,
    nb_init=1,
    iter_max=20,
    nb_threads=1,
    verbose=True
):
    """ K-means algorithm in a general multivariate context with an arbitary
        distance and an arbitray way to choose clusters center:
        Objective is to obtain a partion C = {C_0,..., C_{K-1}} of the data,
        by computing centers mu_i and assigning samples by closest distance.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a _FeatureArray
            * K = number of classes
            * distance = a distance function from class Feature
            * mean_function = a mean computation function from class Feature
            * init = a (N) array with one class per point
            (for example coming from a H-alpha decomposition).
            * nb_init = number of initialisations of K-means
            * eps = stopping threshold
            * iter_max = number of maximum iterations of algorithm
            * nb_threads = number of parallel threads (cores of machine)
            * verbose = bool

        Outputs:
        ---------
            * C = an array of shape (N,) containing labels in {0,..., K-1}
            * mu = an array of shape (p,K) corresponding to classes centers
            * i = number of iterations done
            * delta = convergence criterion
            * criterion_values = list of values of within-classes variances
    """
    assert type(X) == _FeatureArray

    if verbose:
        print('K-means: ' + str(nb_init) + ' init ...')

    if nb_init > 1:
        assert init is None
    t_beginning = time.time()
    best_criterion_value = np.inf
    all_criterion_values = list()
    iterator = range(nb_init)
    if verbose:
        iterator = tqdm(iterator)
    for _ in iterator:
        N = len(X)

        # -------------------------------
        # Initialisation of center means
        # -------------------------------
        if init is None:
            indexes = random_index_for_initialisation(K, N)
            mu = X[indexes]
        else:
            mu = compute_means_parallel(
                X,
                init,
                mean_function,
                nb_threads,
                verbose=verbose
            )

        criterion_value = np.inf
        criterion_values = list()
        delta = np.inf  # Diff between previous and new value of criterion
        i = 0  # Iteration
        C = np.empty(N)  # To store clustering results

        while True:
            # -----------------------------------------
            # Computing distance
            # -----------------------------------------
            d = compute_pairwise_distances_parallel(
                X,
                mu,
                distance,
                nb_threads
            )

            # -----------------------------------------
            # Assigning classes
            # -----------------------------------------
            C = np.argmin(d, axis=1)

            # ---------------------------------------------
            # Managing algorithm convergence
            # ---------------------------------------------
            new_criterion_value = compute_objective_function(d)
            criterion_values.append(new_criterion_value)

            if criterion_value != np.inf:
                delta = np.abs(criterion_value - new_criterion_value)
                delta = delta / criterion_value
                if delta < eps:
                    break
            if (i == iter_max) and (iter_max != 1):
                warnings.warn('K-means algorithm did not converge')
                break

            criterion_value = new_criterion_value

            # -----------------------------------------
            # Computing new means using assigned samples
            # -----------------------------------------
            mu = compute_means_parallel(
                X,
                C,
                mean_function,
                nb_threads,
                verbose=verbose
            )

            i = i + 1

        all_criterion_values.append(criterion_values)

        if criterion_values[-1] < best_criterion_value:
            C_best = C
            mu_best = mu
            i_best = i
            delta_best = delta

    if verbose:
        print('K-means done in %f s.' % (time.time()-t_beginning))

    return C_best, mu_best, i_best, delta_best, all_criterion_values


def K_means(
    X,
    K,
    feature,
    init=None,
    eps=1e-2,
    nb_init=1,
    iter_max=20,
    nb_threads=1,
    verbose=True
):
    """ K-means algorithm.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a numpy Array of shape (batch_size, p, N)
                * batch_size: number of batch
                * p: dimension
                * N: number of vectors for one batch
            * K = number of classes
            * a Feature: for example see pyCovariance/features/covariance.py
            * init = a (batch_size) array with one class per point
            If None, centers are randomly chosen among samples.
            * nb_init = number of initialisations of K-means
            * eps = stopping threshold
            * iter_max = number of maximum iterations of algorithm
            * nb_threads = number of parallel threads (cores of machine)
            * verbose = bool

        Outputs:
        ---------
            * C = an array of shape (N,) containing labels in {0,..., K-1}
            * mu = an array of shape (p,K) corresponding to classes centers
            * i = number of iterations done
            * delta = convergence criterion
            * criterion_values = list of values of within-classes variances
    """
    assert type(X) == np.ndarray
    assert X.ndim == 3
    assert type(feature) == Feature

    if verbose:
        print('########## K-means: ' + str(feature) + ' ##########')
        print('Estimation ...')

    features_array = feature.estimation(X[0])
    if verbose:
        pbar = tqdm(total=X.shape[0])
        pbar.update(1)

    for i in range(1, X.shape[0]):
        features_array.append(feature.estimation(X[i]))
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()

    C, mu, i, delta, criterion_values = _K_means(
        features_array,
        K,
        feature.distance,
        feature.mean,
        init,
        eps,
        nb_init,
        iter_max,
        nb_threads,
        verbose
    )

    return C, mu, i, delta, criterion_values
