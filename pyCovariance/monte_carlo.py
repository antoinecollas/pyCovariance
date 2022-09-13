import autograd.numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def monte_carlo(
    true_parameters,
    sample_fct,
    features_list,
    n_MC,
    n_jobs=1,
    verbose=True
):
    if type(true_parameters) in [list, tuple]:
        errors = np.zeros((len(true_parameters)+1, len(features_list), n_MC))
    else:
        errors = np.zeros((len(features_list), n_MC))

    def _one_MC(
        true_parameters,
        sample_fct,
        features_list
    ):
        X = sample_fct()
        p, N = X.shape
        tmp_f = [f(p, N) for f in features_list]
        errors = list()
        for j, feature in enumerate(tmp_f):
            parameter = feature.estimation(X)
            if type(true_parameters) in [list, tuple]:
                errors.append(feature.distances(parameter, true_parameters))
            else:
                errors.append(feature.distance(parameter, true_parameters))
        errors = np.array(errors)
        return errors

    res = list()
    if n_jobs == 1:
        iterator = range(n_MC)
        if verbose:
            iterator = tqdm(iterator)
        for i in iterator:
            res.append(_one_MC(
                true_parameters,
                sample_fct,
                features_list
            ))
    else:
        res = Parallel(n_jobs=n_jobs)(
            delayed(_one_MC)(
                true_parameters,
                sample_fct,
                features_list
            ) for i in range(n_MC))

    errors = np.stack(res, axis=0)
    errors = errors**2
    mean_errors = np.mean(errors, axis=0)

    return mean_errors
