import autograd.numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
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

        t_beginning = time()

        # features estimation
        X = _estimate_features(X, feature.estimation, n_jobs)

        # centroids computation
        self._means = _compute_means(X, y, feature.mean, n_jobs)

        if verbose:
            print('MDM fitting done in %f s.' % (time() - t_beginning))

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


class TSclassifier(BaseEstimator, ClassifierMixin):

    """Classification in the tangent space.

    Parameters
    ----------
    feature : a Feature from pyCovariance.features
        e.g see pyCovariance/features/covariance.py
    clf: sklearn classifier (default LogisticRegression)
        The classifier to apply in the tangent space.
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
        clf=LogisticRegression(max_iter=int(1e3)),
        n_jobs=1,
        verbose=False
    ):
        """Init."""
        self.base_feature = feature
        self.clf = clf
        self.n_jobs = n_jobs
        self.verbose = verbose

        if not isinstance(clf, ClassifierMixin):
            raise TypeError('clf must be a ClassifierMixin')

    def fit(self, X, y):
        """Fit TSclassifier.
        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)
        y : ndarray shape (n_samples,)

        Returns
        -------
        self : TSclassifier instance.
        """
        verbose = self.verbose
        n_jobs = self.n_jobs

        n_samples, p, N = X.shape
        feature = self.feature = self.base_feature(p, N)

        clf = self.clf

        if verbose:
            print('Feature: ' + str(feature))
            print('TSclassifier fitting ...')

        t_beginning = time()

        # features estimation
        X = _estimate_features(X, feature.estimation, n_jobs)

        # centroid computation
        mean = self._mean = _compute_means(
            X,
            y=np.zeros((n_samples)),
            mean_fct=feature.mean,
            n_jobs=n_jobs
        )

        # log of X at self._mean
        X = feature.log(mean, X, vectorize=True)

        # classifier fitting
        self.clf = clf.fit(X, y)

        if verbose:
            print('TSclassifier fitting done in %f s.'
                  % (time() - t_beginning))

        return self

    def predict(self, X):
        """get the predictions.
        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)

        Returns
        -------
        pred : ndarray of int, shape (n_trials,)
        """
        feature = self.feature
        mean = self._mean
        n_jobs = self.n_jobs
        clf = self.clf

        # features estimation
        X = _estimate_features(X, feature.estimation, n_jobs)

        # log of X at self._mean
        X = feature.log(mean, X, vectorize=True)

        # classifier prediction
        pred = clf.predict(X)

        return pred

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """get the probabilities.
        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
        """
        feature = self.feature
        mean = self._mean
        n_jobs = self.n_jobs
        clf = self.clf

        # features estimation
        X = _estimate_features(X, feature.estimation, n_jobs)

        # log of X at self._mean
        X = feature.log(mean, X, vectorize=True)

        # classifier prediction
        pred = clf.predict_proba(X)

        return pred
