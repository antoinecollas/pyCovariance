import autograd.numpy as np
from tqdm import tqdm


def monte_carlo(
    true_parameters,
    sample_fct,
    features_list,
    nb_MC,
    verbose=True
):
    if type(true_parameters) in [list, tuple]:
        errors = np.zeros((len(true_parameters)+1, len(features_list), nb_MC))
    else:
        errors = np.zeros((len(features_list), nb_MC))

    iterator = range(nb_MC)
    if verbose:
        iterator = tqdm(iterator)
    for i in iterator:
        X = sample_fct()
        p, N = X.shape
        tmp_f = [f(p, N) for f in features_list]
        for j, feature in enumerate(tmp_f):
            parameter = feature.estimation(X)
            if type(true_parameters) in [list, tuple]:
                errors[:, j, i] = feature.distances(parameter, true_parameters)
            else:
                errors[j, i] = feature.distances(parameter, true_parameters)[0]

    errors = errors**2
    mean_errors = np.mean(errors, axis=-1)

    return mean_errors
