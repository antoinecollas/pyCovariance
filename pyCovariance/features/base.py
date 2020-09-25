import autograd.numpy as np
from functools import partial

class BaseClassFeatures:
    def __init__(self):
        pass
    
    def __str__(self):
        raise NotImplementedError

    def estimation(self, X):
        """ Serve to compute feature.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation

        Outputs:
        ---------
            * feature = a (feature_size) array
        """
        raise NotImplementedError

    def distance(self, x_1, x_2):
        """ Compute distance between two features.
            ----------------------------------------------------------------------
            Inputs:
            --------
                * x_1 = feature n°1
                * x_2 = feature n°2
            Outputs:
            ---------
                * distance = a scalar
            """
        raise NotImplementedError

    def mean(self, X):
        """ Compute mean of features
            ----------------------------------------------------------------------
            Inputs:
            --------
                * X = array of shape (feature_size, M) corresponding to samples in class
            Outputs:
            ---------
                * mean = a (feature_size) array
            """
        raise NotImplementedError


def center_vectors(X):
    """ Serve to center vectors (e.g pixels).
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number of samples used for estimation

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """
    mean = np.mean(X, axis=1)
    mean = mean[:, np.newaxis]
    X = X - mean
    return X


def center_vectors_estimation(features):
    """ Center vectors before estimating features.
    -------------------------------------
        Inputs:
            --------
            * features = class of type BaseClassFeatures (e.g CovarianceTexture) 
        Outputs:
            ---------
            * the same class as in the input except the vectors are now centered before estimating
    """
    def estimation_on_centered_vectors(X, features_estimation):
        X = center_vectors(X)
        return features_estimation(X)
    features.estimation = partial(estimation_on_centered_vectors, features_estimation=features.estimation)
    return features
