from functools import partial
import numpy as np

# import functions related to covariance features
from .covariance_clustering_functions import center_vectors, distance_covariance_Euclidean, distance_covariance_Riemannian, mean_covariance_Euclidean, mean_covariance_Riemannian, vech_SCM

# import functions related to covariance and texture features
from .covariance_and_texture_clustering_functions import compute_feature_covariance_texture, distance_covariance_texture_Riemannian, mean_covariance_texture_Riemannian

# import functions related to location and covariance features
from .location_covariance_clustering_functions import compute_feature_location_covariance, distance_location_covariance_Euclidean, mean_location_covariance_Euclidean


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


class PixelEuclidean(BaseClassFeatures):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return 'Pixel_Euclidean'
    
    def estimation(self, X):
        center_pixel = X[:, X.shape[1]//2+1, :]
        return center_pixel

    def distance(self, x1, x2):
        d = np.linalg.norm(x2-x1)
        d = np.real(d)
        return d

    def mean(self, X):
        return np.mean(X, axis=1)


class MeanPixelEuclidean(BaseClassFeatures):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return 'Mean_Pixel_Euclidean'
    
    def estimation(self, X):
        return np.mean(X, 1)

    def distance(self, x1, x2):
        d = np.linalg.norm(x2-x1)
        d = np.real(d)
        return d

    def mean(self, X):
        return np.mean(X, axis=1)


class Intensity(BaseClassFeatures):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return 'Intensity'
    
    def estimation(self, X):
        center_pixel = X[:, X.shape[1]//2+1, :]
        intensity = np.linalg.norm(center_pixel)
        return intensity

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
        return 'Covariance_Euclidean'
    
    def estimation(self, X):
        return vech_SCM(X)

    def distance(self, x1, x2):
        return distance_covariance_Euclidean(x1, x2)

    def mean(self, X):
        return mean_covariance_Euclidean(X)


class Covariance(BaseClassFeatures):
    def __init__(
        self,
        mean_args=None
    ):
        super().__init__()
        self.mean_args = mean_args
    
    def __str__(self):
        return 'Covariance_Riemannian'

    def estimation(self, X):
        return vech_SCM(X)

    def distance(self, x1, x2):
        return distance_covariance_Riemannian(x1, x2)

    def mean(self, X):
        if self.mean_args:
            return mean_covariance_Riemannian(X, self.mean_args)
        return mean_covariance_Riemannian(X)

class CovarianceTexture(BaseClassFeatures):
    def __init__(
        self,
        p,
        N,
        estimation_args=None,
        mean_args=None
    ):
        super().__init__()
        self.p = p
        self.N = N
        distance_args =  (p, N)
        self.estimation_args = estimation_args
        self.mean_args = mean_args
    
    def __str__(self):
        return 'Covariance_texture_Riemannian'
    
    def estimation(self, X):
        if self.estimation_args is not None:
            return compute_feature_covariance_texture(X, self.estimation_args)
        return compute_feature_covariance_texture(X)

    def distance(self, x1, x2):
        return distance_covariance_texture_Riemannian(x1, x2, self.p, self.N)

    def mean(self, X):
        if self.mean_args:
            return mean_covariance_texture_Riemannian(X, self.p, self.N, self.mean_args)
        return mean_covariance_texture_Riemannian(X, self.p, self.N)

class LocationCovarianceEuclidean(BaseClassFeatures):
    def __init__(
        self,
        p
    ):
        super().__init__()
        self.p = p
    
    def __str__(self):
        return 'Location_And_Covariance_Euclidean'

    def estimation(self, X):
        return compute_feature_location_covariance(X)

    def distance(self, x1, x2):
        return distance_location_covariance_Euclidean(x1, x2, self.p)

    def mean(self, X):
        return mean_location_covariance_Euclidean(X)


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
