import autograd.numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax


def _estimate_features(X, estimation_fct, n_jobs=1):
    if n_jobs == 1:
        temp = [estimation_fct(X[i]) for i in range(len(X))]
    else:
        temp = Parallel(n_jobs=n_jobs)(
            delayed(estimation_fct)(X[i]) for i in range(len(X)))

    X = temp[0]
    for t in temp[1:]:
        X.append(t)

    return X


def _compute_means(X, y, mean_fct, n_jobs=1):
    classes = np.unique(y)
    if n_jobs == 1:
        temp = [mean_fct(X[y == i]) for i in classes]
    else:
        temp = Parallel(n_jobs=n_jobs)(
            delayed(mean_fct)(X[y == i]) for i in classes)

    means = temp[0]
    for m in temp[1:]:
        means.append(m)

    return means


def _compute_pairwise_distances(X, means, distance_fct, n_jobs=1):

    def _compute_distances_to_mean(X, mean, distance_fct):
        distances = np.zeros((len(X)))
        for j in range(len(X)):
            distances[j] = distance_fct(X[j], mean)
        return distances

    if n_jobs == 1:
        distances = np.zeros((len(X), len(means)))
        for i in range(len(means)):
            distances[:, i] = _compute_distances_to_mean(
                X, means[i], distance_fct)
    else:
        temp = Parallel(n_jobs=n_jobs)(
            delayed(_compute_distances_to_mean)(X, means[i], distance_fct)
            for i in range(len(means)))
        distances = np.stack(temp, axis=1)

    return distances


class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean
    using covariance estimation and Riemannian geometry.

    Parameters
    ----------
    feature : a Feature from pyCovariance.features
        e.g see pyCovariance/features/covariance.py
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    verbose: bool
    """

    def __init__(
        self,
        feature,
        n_jobs=1,
        verbose=False
    ):
        self.base_feature = feature
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """Estimate features and centroids.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)
        y : ndarray shape (n_samples, 1)

        Returns
        -------
        self : MDM instance.
        """
        n_jobs = self.n_jobs
        verbose = self.verbose

        p, N = X.shape[1:]
        feature = self.feature = self.base_feature(p, N)

        if verbose:
            print('Feature: ' + str(feature))
            print('MDM fitting ...')

        self._classes = np.unique(y)

        # features estimation
        X = _estimate_features(X, feature.estimation, n_jobs)

        # centroids computation
        self._means = _compute_means(X, y, feature.mean, n_jobs)

        return self

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)

        Returns
        -------
        pred : ndarray of int, shape (n_samples, 1)
            the prediction for each sample according to the closest centroid.
        """
        distances = self.transform(X)
        return self._classes[distances.argmin(axis=1)]

    def transform(self, X):
        """get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)

        Returns
        -------
        dist : ndarray, shape (n_samples, n_classes)
            the distance to each centroid according to the metric.
        """
        n_jobs = self.n_jobs
        feature = self.feature
        means = self._means

        # features estimation
        X = _estimate_features(X, feature.estimation, n_jobs)

        # centroids computation
        distances = _compute_pairwise_distances(
            X, means, feature.distance, n_jobs)

        return distances

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

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
        return softmax(-self.transform(X))
