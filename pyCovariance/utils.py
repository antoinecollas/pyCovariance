import autograd.numpy as np
from joblib import Parallel, delayed


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
