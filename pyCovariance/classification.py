import autograd.numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax


class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean.

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

    """

    def __init__(self, feature, n_jobs=1):
        self.base_feature = feature
        self.n_jobs = n_jobs

    @staticmethod
    def _estimate_features(feature, X, n_jobs=1):
        if n_jobs == 1:
            temp = feature.estimation(X[0])
            for i in range(1, len(X)):
                temp.append(feature.estimation(X[i]))
            X = temp
        return X

    @staticmethod
    def _compute_centroids(feature, X, y, n_jobs=1):
        classes = np.unique(y)
        if n_jobs == 1:
            temp = [feature.mean(X[y == i]) for i in classes]
        #else:
            #self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
            #    delayed(mean_covariance)(X[y == l], metric=self.metric_mean,
            #                             sample_weight=sample_weight[y == l])
            #    for l in self.classes_)
        means = temp[0]
        for m in temp[1:]:
            means.append(m)
        return means

    @staticmethod
    def _compute_pairwise_distances(feature, X, means, n_jobs=1):
        distances = np.zeros((len(X), len(means)))
        if n_jobs == 1:
            for i in range(len(means)):
                for j in range(len(X)):
                    distances[j, i] = feature.distance(X[j], means[i])
        return distances

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

        p, N = X.shape[1:]
        feature = self.feature = self.base_feature(p, N)

        self._classes = np.unique(y)
        classes = self._classes

        # features estimation
        X = self._estimate_features(feature, X, n_jobs)

        # centroids computation
        self._means = self._compute_centroids(feature, X, y, n_jobs)

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
        classes = self._classes
        means = self._means

        # features estimation
        X = self._estimate_features(feature, X, n_jobs)

        # centroids computation
        distances = self._compute_pairwise_distances(feature, X, means, n_jobs)

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
