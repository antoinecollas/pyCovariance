# import functions related to covariance features
from .covariance_clustering_functions import vech_SCM, covariance_arithmetic_mean, covariance_Euclidean_distance, Riemannian_distance_covariance, Riemannian_mean_covariance

# import functions related to covariance and texture features
from .covariance_and_texture_clustering_functions import compute_feature_Covariance_texture, Riemannian_distance_covariance_texture, Riemannian_mean_covariance_texture

class BaseClassFeatures:
    def __init__(
        self,
        estimation_parameters=None,
        distance_parameters=None,
        mean_parameters=None
    ):
        self.estimation_parameters = estimation_parameters
        self.distance_parameters = distance_parameters
        self.mean_parameters = mean_parameters
    
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
        mean_parameters=[1.0, 0.95, 1e-9, 5, False, 0]
    ):
        super().__init__(
            mean_parameters=mean_parameters
        )
    
    def __str__(self):
        return 'Covariance_Riemannian_features'

    def estimation(self, X):
        return vech_SCM(X)

    def distance(self, x1, x2):
        return Riemannian_distance_covariance(x1, x2)

    def mean(self, X):
        return Riemannian_mean_covariance(X, self.mean_parameters)

class CovarianceTexture(BaseClassFeatures):
    def __init__(
        self,
        p,
        N,
        estimation_parameters=(0.01, 20),
        mean_parameters=[1.0, 0.95, 1e-9, 5, False, 0]
    ):
        distance_parameters =  (p, N)
        mean_parameters =  [p, N] + mean_parameters
        super().__init__(
            estimation_parameters=estimation_parameters,
            distance_parameters=distance_parameters,
            mean_parameters=mean_parameters
        )
    
    def __str__(self):
        return 'Covariance_texture_Riemannian_features'
    
    def estimation(self, X):
        return compute_feature_Covariance_texture(X, self.estimation_parameters)

    def distance(self, x1, x2):
        return Riemannian_distance_covariance_texture(x1, x2, self.distance_parameters)

    def mean(self, X):
        return Riemannian_mean_covariance_texture(X, self.mean_parameters)
