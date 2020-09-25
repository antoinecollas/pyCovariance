import autograd.numpy as np

from .base import BaseClassFeatures
from ..vectorization import *

########## ESTIMATION ##########

def compute_feature_location_covariance(X):
    """ Serve to compute feature for mean and Covariance classification.
        We use vech opeartion to save memory space.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number of samples used for estimation

        Outputs:
        ---------
            * the feature for classification
        """
    X = np.squeeze(X)
    m = mean(X)[:, np.newaxis]
    X = X - m
    cov = vech(SCM(X))
    mean_cov = np.hstack([m.squeeze(),cov])
    return mean_cov


##########  DISTANCE  ##########

def distance_location_covariance_Euclidean(x1, x2, p):
    """ Serve to compute distance between two (\mu, \Sigma).
        We use vech opeartion to save memory space.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * x1 = a (p+p*(p+1)/2,) array where p is the dimension of data
            * x2 = a (p+p*(p+1)/2,) array where p is the dimension of data
            * p = the dimension of data

        Outputs:
        ---------
            * distance
        """
    mu_1 = x1[:p]
    mu_2 = x2[:p]
    sigma_1 = unvech(x1[p:])
    sigma_2 = unvech(x2[p:])
    d_mu = np.real(np.linalg.norm(mu_2-mu_1))**2
    d_sigma = np.real(np.linalg.norm(sigma_2-sigma_1))**2
    d = np.sqrt(d_mu + d_sigma)
    return d


##########   MEAN     ##########

def mean_location_covariance_Euclidean(X_class):
    """ Euclidean mean on location and covariance
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X_class = array of shape (p+p*(p+1)/2, N) corresponding to 
                        samples in class

        Outputs:
        ---------
            * mu = the vech of Euclidean mean
        """
    m = np.mean(X_class, axis=1)
    return m


##########  CLASSES  ##########

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