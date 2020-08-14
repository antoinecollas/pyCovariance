from functools import partial
import numpy as np

# import functions related to covariance features
from .covariance_clustering_functions import center_vectors, covariance_arithmetic_mean, covariance_Euclidean_distance, Riemannian_distance_covariance, Riemannian_mean_covariance, vech_SCM

# import functions related to covariance and texture features
from .covariance_and_texture_clustering_functions import compute_feature_Covariance_texture, Riemannian_distance_covariance_texture, Riemannian_mean_covariance_texture

class BaseClassFeatures:
    def __init__(
        self,
        estimation_args=None,
        distance_args=None,
        mean_args=None
    ):
        self.estimation_args = estimation_args
        self.distance_args = distance_args
        self.mean_args = mean_args
    
    def __str__(self):
        return self.__str__()

    def vec(self, feature):
        """ Serve to vectorize a feature. (For example, it vectorizes a matrix of covariance).
        ----------------------------------------------------------------------
        Inputs:
        --------
            * feature = an array

        Outputs:
        ---------
            * feature = a (feature_size) array
        """
        raise NotImplementedError

    def unvec(self, feature):
        """ Serve to un-vectorize a feature. It is the reverse operation of the vec method.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * feature = a (feature_size) array

        Outputs:
        ---------
            * feature = an array
        """
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

class PixelEuclidean(BaseClassFeatures):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return 'Pixel_Euclidean_features'
    
    def estimation(self, X):
        center_pixel = X[:, X.shape[1]//2+1, :]
        return center_pixel

    def distance(self, x1, x2):
        d = np.linalg.norm(x2-x1)
        d = np.real(d)
        return d

    def mean(self, X):
        return np.mean(X, axis=1)

class CovarianceEuclidean(BaseClassFeatures):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return 'Covariance_Euclidean_features'
    
    def estimation(self, X):
        return vech_SCM(X)

    def distance(self, x1, x2):
        return covariance_Euclidean_distance(x1, x2)

    def mean(self, X):
        return covariance_arithmetic_mean(X)

class Covariance(BaseClassFeatures):
    def __init__(
        self,
        mean_args=[1.0, 0.95, 1e-9, 5, False, 0]
    ):
        super().__init__(
            mean_args=mean_args
        )
    
    def __str__(self):
        return 'Covariance_Riemannian_features'

    def estimation(self, X):
        return vech_SCM(X)

    def distance(self, x1, x2):
        return Riemannian_distance_covariance(x1, x2)

    def mean(self, X):
        return Riemannian_mean_covariance(X, self.mean_args)

class CovarianceTexture(BaseClassFeatures):
    def __init__(
        self,
        p,
        N,
        estimation_args=(0.01, 20),
        mean_args=[1.0, 0.95, 1e-9, 5, False, 0]
    ):
        distance_args =  (p, N)
        mean_args =  [p, N] + mean_args
        super().__init__(
            estimation_args=estimation_args,
            distance_args=distance_args,
            mean_args=mean_args
        )
    
    def __str__(self):
        return 'Covariance_texture_Riemannian_features'
    
    def estimation(self, X):
        return compute_feature_Covariance_texture(X, self.estimation_args)

    def distance(self, x1, x2):
        return Riemannian_distance_covariance_texture(x1, x2, self.distance_args)

    def mean(self, X):
        return Riemannian_mean_covariance_texture(X, self.mean_args)

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
