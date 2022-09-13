import autograd.numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from time import time

from .utils import\
        _compute_means,\
        _compute_pairwise_distances,\
        _estimate_features


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
        y : ndarray shape (n_samples,)

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

        t1 = time()

        # features estimation
        self._features_X_train = X = _estimate_features(
            X, feature.estimation, n_jobs)

        if verbose:
            print('MDM: estimation done in %f s.' % (time() - t1))

        t2 = time()

        # centroids computation
        self._means = _compute_means(X, y, feature.mean, n_jobs=n_jobs)
        if verbose:
            print('MDM: centroid computation done in %f s.' % (time() - t2))
            print('MDM fitting done in %f s.' % (time() - t1))

        return self

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)

        Returns
        -------
        pred : ndarray of int, shape (n_samples,)
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
        verbose = self.verbose

        t1 = time()

        # features estimation
        X = _estimate_features(X, feature.estimation, n_jobs)

        if verbose:
            print('MDM: estimation done in %f s.' % (time() - t1))

        t2 = time()

        # distances computation
        distances = _compute_pairwise_distances(
            X, means, feature.distance, n_jobs=n_jobs)

        if verbose:
            print('MDM: distances computation done in %f s.' % (time() - t2))

        return distances

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        verbose = self.verbose

        self.fit(X, y)

        t1 = time()

        # distances computation
        distances = _compute_pairwise_distances(
            self._features_X_train, self._means,
            self.feature.distance, n_jobs=self.n_jobs)

        if verbose:
            print('MDM: distances computation done in %f s.' % (time() - t1))

        return self._classes[distances.argmin(axis=1)]

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
