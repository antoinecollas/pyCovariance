import autograd.numpy as np
from tqdm import tqdm


def monte_carlo(true_parameters, sample_fct, features_list, nb_MC):

    if type(true_parameters) in [list, tuple]:
        errors = np.zeros((len(true_parameters)+1, len(features_list), nb_MC))
    else:
        errors = np.zeros((len(features_list), nb_MC))

    for i in tqdm(range(nb_MC)):
        X = sample_fct()
        for j, feature in enumerate(features_list):
            parameter = feature.estimation(X)
            if type(true_parameters) in [list, tuple]:
                errors[:, j, i] = feature.distances(parameter, true_parameters)
            else:
                errors[j, i] = feature.distances(parameter, true_parameters)[0]

    errors = errors**2
    mean_errors = np.mean(errors, axis=-1)

    return mean_errors
