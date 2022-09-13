import autograd.numpy as np
from sklearn.base import\
        BaseEstimator,\
        ClassifierMixin,\
        TransformerMixin,\
        ClusterMixin
from time import time
from tqdm import tqdm
import warnings

from .classification import\
        MDM
from .features.base import _FeatureArray
from .utils import\
        _estimate_features,\
        _compute_means,\
        _compute_pairwise_distances


def _compute_objective_function(distances):
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


def _random_index_for_initialisation(n_clusters, n_samples):
    indexes = list()
    for k in range(n_clusters):
        index = np.random.randint(n_samples)
        while index in indexes:
            index = np.random.randint(n_samples)
        indexes.append(index)
    return indexes


def _init_K_means_plus_plus(
    X,
    n_clusters,
    distance,
    n_jobs=1
):
    indexes = list()
    indexes.append(np.random.randint(len(X)))
    for k in range(1, n_clusters):
        mu = X[indexes]
        d = _compute_pairwise_distances(
            X,
            mu,
            distance,
            n_jobs
        )
        d = d**2
        d = np.min(d, axis=1)
        d = d / np.sum(d)
        index = np.argmax(d)
        indexes.append(index)
    return indexes


def _K_means(
    X,
    n_clusters,
    distance,
    mean_function,
    init='k-means++',
    tol=1e-2,
    n_init=1,
    max_iter=20,
    n_jobs=1,
    verbose=True
):
    """ K-means algorithm in a general multivariate context.
        Objective is to obtain a partion C = {C_0,..., C_{K-1}} of the data,
        by computing centers and assigning samples by closest distance.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a _FeatureArray
            * n_clusters = number of classes
            * distance = a distance function from class Feature
            * mean_function = a mean computation function from class Feature
            * init = 'random' or 'k-means++'
            * tol = stopping threshold
            * n_init = number of initialisations of K-means
            * max_iter = number of maximum iterations of algorithm
            * n_jobs = number of parallel threads (cores of machine)
            * verbose = bool

        Outputs:
        ---------
            * C = an array of shape (N,) containing labels in {0,..., K-1}
            * mu = an array of shape (p, n_clusters)
            corresponding to classes centers
            * i = number of iterations done
            * delta = convergence criterion
            * criterion_values = list of values of within-classes variances
    """
    assert type(X) == _FeatureArray
    assert init in ['random', 'k-means++']

    if verbose:
        if init == 'random':
            print('K-means: ' + str(n_init) + ' init ...')
        elif init == 'k-means++':
            print('K-means++: ' + str(n_init) + ' init ...')

    t_beginning = time()
    best_criterion_value = np.inf
    all_criterion_values = list()
    iterator = range(n_init)
    if verbose:
        iterator = tqdm(iterator)
    for _ in iterator:
        N = len(X)

        # -------------------------------
        # Initialisation of center means
        # -------------------------------
        if init == 'random':
            indexes = _random_index_for_initialisation(n_clusters, N)
        elif init == 'k-means++':
            indexes = _init_K_means_plus_plus(
                X,
                n_clusters,
                distance,
                n_jobs
            )
        mu = X[indexes]

        criterion_value = np.inf
        criterion_values = list()
        delta = np.inf  # Diff between previous and new value of criterion
        i = 1  # Iteration
        C = np.empty(N)  # To store clustering results

        while True:
            # -----------------------------------------
            # Computing distance
            # -----------------------------------------
            d = _compute_pairwise_distances(
                X,
                mu,
                distance,
                n_jobs=n_jobs
            )

            # -----------------------------------------
            # Assigning classes
            # -----------------------------------------
            C = np.argmin(d, axis=1)

            # ---------------------------------------------
            # Managing algorithm convergence
            # ---------------------------------------------
            new_criterion_value = _compute_objective_function(d)
            criterion_values.append(new_criterion_value)

            if criterion_value != np.inf:
                delta = np.abs(criterion_value - new_criterion_value)
                delta = delta / criterion_value
                if delta < tol:
                    break
            if (i == max_iter) and (max_iter != 1):
                warnings.warn('K-means algorithm did not converge')
            if i == max_iter:
                break

            criterion_value = new_criterion_value

            # -----------------------------------------
            # Computing new means using assigned samples
            # -----------------------------------------
            mu = _compute_means(
                X,
                C,
                mean_function,
                n_jobs=n_jobs
            )

            i = i + 1

        all_criterion_values.append(criterion_values)

        if criterion_values[-1] < best_criterion_value:
            best_criterion_value = criterion_values[-1]
            C_best = C
            mu_best = mu
            i_best = i
            delta_best = delta

    if verbose:
        print('K-means done in %f s.' % (time()-t_beginning))

    return C_best, mu_best, i_best, delta_best, all_criterion_values


class K_means(BaseEstimator, ClassifierMixin, ClusterMixin, TransformerMixin):

    """K-means clustering using covariance estimation and Riemannian geometry.

    Find clusters that minimize the sum of squared distance to their centroid.

    Parameters
    ----------
    feature : a Feature from pyCovariance.features
        e.g see pyCovariance/features/covariance.py
    n_clusters: int (default: 2)
        number of clusters.
    init : 'random' or 'k-means++'
    max_iter : int (default: 100)
        The maximum number of iteration to reach convergence.
    n_init : int, (default: 10)
        Number of time the k-means algorithm will be run with different
        initialisation of centroids. The final results will be
        the best output of n_init consecutive runs in terms of inertia.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    tol: float, (default: 1e-4)
        the stopping criterion to stop convergence, representing the minimum
        amount of change in labels between two iterations.
    verbose: bool
    """

    def __init__(
        self,
        feature,
        n_clusters=2,
        init='k-means++',
        max_iter=100,
        n_init=10,
        n_jobs=1,
        tol=1e-4,
        verbose=False
    ):
        assert init in ['random', 'k-means++']
        self.base_feature = feature
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.n_jobs = n_jobs
        self.verbose = verbose

        # init MDM
        self._mdm = MDM(feature=feature, n_jobs=n_jobs, verbose=False)
        self._mdm._classes = np.arange(n_clusters)

    def fit(self, X, y=None):
        """Estimate features and cluster them.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Kmeans instance.
        """
        p, N = X.shape[1:]
        feature = self.feature = self.base_feature(p, N)
        n_clusters = self.n_clusters
        init = self.init
        max_iter = self.max_iter
        n_init = self.n_init
        tol = self.tol
        n_jobs = self.n_jobs
        verbose = self.verbose

        # estimate features
        X = _estimate_features(X, feature.estimation, n_jobs=n_jobs)
        if verbose:
            print('Feature: ' + str(feature))

        y_pred_train, centroids, _, _, criterion_values = _K_means(
            X,
            n_clusters,
            feature.distance,
            feature.mean,
            init=init,
            tol=tol,
            n_init=n_init,
            max_iter=max_iter,
            n_jobs=n_jobs,
            verbose=verbose
        )

        self._y_pred_train = y_pred_train
        self._criterion_values = criterion_values
        self._mdm.feature = feature
        self._mdm._means = centroids

        return self

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        return self._mdm.predict(X)

    def transform(self, X):
        """get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)

        Returns
        -------
        dist : ndarray, shape (n_trials, n_cluster)
            the distance to each centroid according to the metric.
        """
        return self._mdm.transform(X)

    def fit_predict(self, X, y=None):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self._y_pred_train

    def predict_proba(self, X):
        """Predict proba using softmax.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)

        Returns
        -------
        prob : ndarray, shape (n_samples, n_classes)
            the softmax probabilities for each class.
        """
        return self._mdm.predict_proba(X)
