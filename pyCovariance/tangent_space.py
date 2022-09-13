import autograd.numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from time import time

from .utils import\
        _compute_means,\
        _estimate_features


class TangentSpace(BaseEstimator, TransformerMixin):

    """Estimates covariance matrices and then lifts them into tangent space.

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
        """Init."""
        self.base_feature = feature
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        """Estimates features and centroid of features.
        This centroid will be used in the Riemannian log to lift features
        in tangent space.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)
        y : None.
            y is not used, only for compatibility with sklearn.

        Returns
        -------
        self : TangentSpace instance.
        """
        verbose = self.verbose
        n_jobs = self.n_jobs

        n_samples, p, N = X.shape
        feature = self.feature = self.base_feature(p, N)

        if verbose:
            print('Feature: ' + str(feature))
            print('TangentSpace fitting ...')

        t_beginning = time()

        # features estimation
        X = _estimate_features(X, feature.estimation, n_jobs)

        # centroid computation
        self._mean = _compute_means(
            X,
            y=np.zeros((n_samples)),
            mean_fct=feature.mean,
            n_jobs=n_jobs
        )

        if verbose:
            print('TangentSpace fitting done in %f s.'
                  % (time() - t_beginning))

        return self

    def transform(self, X):
        """Estimates features and lifts them in tangent space.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p, N)

        Returns
        -------
        X : ndarray, shape (n_samples, dim)
            dim is the dimension of the manifold.
        """
        verbose = self.verbose
        n_jobs = self.n_jobs
        feature = self.feature
        mean = self._mean

        if verbose:
            print('Feature: ' + str(feature))
            print('TangentSpace transforming ...')

        t_beginning = time()

        # features estimation
        X = _estimate_features(X, feature.estimation, n_jobs)

        # log of X at self._mean
        X = feature.log(mean, X, vectorize=True)

        if verbose:
            print('TangentSpace transforming done in %f s.'
                  % (time() - t_beginning))

        return X
