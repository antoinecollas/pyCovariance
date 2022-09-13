import autograd.numpy as np
import autograd.numpy.random as rnd
from sklearn.metrics import accuracy_score

from pyCovariance.classification import\
        MDM
from pyCovariance.features import\
        center_euclidean
from pyCovariance.generation_data import\
        generate_covariance,\
        sample_normal_distribution
from pyCovariance.testing import assert_allclose


def helper_test_MDM(clf):
    n_classes = 5
    size_class = 100
    n_datasets = 3
    n_samples = size_class * n_classes

    for j in range(n_datasets):
        # test several p, N
        N = 21 + 5*j
        p = 3 + j

        X = np.zeros((n_samples, p, N))

        y = list()
        for k in range(n_classes):
            cov = generate_covariance(p)
            mu = 5*rnd.randn(p, 1)
            for i in range(size_class):
                X[i + k*size_class] = sample_normal_distribution(N, cov) + mu

            y.append(k*np.ones(size_class))
        y = np.concatenate(y)
        assert X.shape[0] == len(y)

        idx = rnd.permutation(n_samples)

        # generating training data
        X_train = X[idx[:n_samples//2]]
        y_train = y[idx[:n_samples//2]]

        # generating test data
        X_test = X[idx[n_samples//2:]]
        y_test = y[idx[n_samples//2:]]

        # fit + predict on train set
        clf.fit(X_train, y_train)
        means = clf._means
        for i, mean in enumerate(means):
            temp = X_train[y_train == i]
            temp = temp[:, :, N//2]
            temp = np.mean(temp, axis=0)
            assert_allclose(mean.export(), temp)
        y = clf.predict(X_train)
        assert accuracy_score(y_train, y) > 0.95

        # fit + predict on test set
        y = clf.predict(X_test)
        assert accuracy_score(y_test, y) > 0.95

        # fit + predict on test set
        y = clf.transform(X_test)
        y = y.argmin(axis=1)
        assert accuracy_score(y_test, y) > 0.95

        # predict proba
        proba = clf.predict_proba(X_test)
        assert (proba >= 0).all()
        actual = np.sum(proba, axis=1)
        desired = np.ones(len(X_test))
        assert ((actual - desired) < 1e-8).all()
        actual = proba.argmax(axis=1)
        assert (actual == y).all()

        # fit_predict
        y = clf.fit_predict(X_train, y_train)
        assert accuracy_score(y_train, y) > 0.95


def test_MDM_single_job():
    rnd.seed(123)

    feature = center_euclidean()
    mdm = MDM(feature, n_jobs=1)
    helper_test_MDM(mdm)


def test_MDM_multiple_jobs():
    rnd.seed(123)

    feature = center_euclidean()
    mdm = MDM(feature, n_jobs=-1)
    helper_test_MDM(mdm)
