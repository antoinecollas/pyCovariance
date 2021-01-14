import autograd.numpy as np
from tqdm import tqdm


def monte_carlo(true_parameter, sample_fct, features_list, nb_MC):
    errors = np.zeros((len(features_list), nb_MC))

    for i in tqdm(range(nb_MC)):
        X = sample_fct()
        for j, feature in enumerate(features_list):
            parameter = feature.estimation(X)
            errors[j, i] = feature.distance(parameter, true_parameter)**2

    mean_errors = np.mean(errors, axis=1)

    return mean_errors
