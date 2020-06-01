import numpy as np
import scipy as sp
import warnings

from .clustering_functions import wrapper_compute_all_mean_parallel
from .covariance_clustering_functions import vech_SCM
from .generic_functions import * 
from .multivariate_images_tools import *

def assign_class_H_α(H, α):
    """ A simple function to assign a class in the H-α plane"""

    if H <= 0.5:
      if α <= 42.5:
        c = 7
      elif α <= 47.5:
        c = 6
      elif α <= 90:
        c = 5
    elif H <= 0.9:
      if α <= 40:
        c = 4
      elif α <= 50:
        c = 3
      elif α <= 90:
        c = 2
    elif H <= 1.0:
      if α <= 55:
        c = 1
      elif α <= 90:
        c = 0
    else:
        c = np.nan

    return c


def compute_h_alpha_class(𝐗, args=None):
    """ Compute H-α values given data in input and assign to a class
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation of covariance matrix
            * args is unused but needed for coherent coding

        Outputs:
        ---------
            * the class corresponding to a zone in the H-α plane
    """

    𝝨 = SCM(np.squeeze(𝐗))
    𝛌, 𝐔 = sp.linalg.eigh(𝝨)
    𝛌 = 𝛌/np.sum(𝛌)
    α_vector = np.arccos(np.abs(𝐔[0,:]))

    H = - np.dot(𝛌, np.log(𝛌))
    α = np.dot(𝛌, α_vector) * (180.0/np.pi)

    return [assign_class_H_α(H, α)]


def cluster_image_by_H_alpha(image, windows_mask, multi=False, 
                     number_of_threads_rows=4, number_of_threads_columns=4):
    """ Basic clustering of a SAR image using the values in the H-α plane
        ----------------------------------------------------------------------
        Inputs:
        --------
            * image = a (n_r, n_c, p) numpy array with:
                * n_r = number of rows of image
                * n_c = number of columns of image
                * p = 3. HH, Hv and VV polarisations in this order
            * windows_mask = a boolean (m_r, m_c) array which is a mask to compute 
                             covariance matrix using SCM
            * enable_multi = enable or not parallel compuation
            * number_of_threads_columns = number of thread to use in columns 
                (total threads = number of cores of the machine in general)
            * number_of_threads_rows = number of thread to use in columns 
                (total threads = number of cores of the machine in general) 

        Outputs:
        ---------
            * 𝓒 = an array of shape (n_r-m_r+1, n_c-m_c+1) corresponding to the 
                  clustering, the borders are not kept since we can't compute SCM
    """

    n_r, n_c, p = image.shape
    m_r, m_c = windows_mask.shape
    𝓒 = sliding_windows_treatment_image_time_series_parallel(image.reshape(n_r,n_c,p,1), 
                     windows_mask, compute_h_alpha_class, None, multi=multi, 
                     number_of_threads_rows=number_of_threads_rows,
                     number_of_threads_columns=number_of_threads_columns)

    return 𝓒.reshape(n_r-m_r+1, n_c-m_c+1)


def initialise_H_alpha(𝐗, init_parameters):
    """ Initialise the mean of classes whatever the features using a H-α 
        initialsiation of classes
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
            * init_parameters = [𝓒, mean_function, mean_parameters, enable_multi] where,
                * 𝓒 is the H-α pre-clustering
                * mean_function = function to compute mean
                              takes two arguments:
                              ** 𝐗_class = array of shape (p, M) corresponding to 
                                           samples in class
                              ** mean_parameters = parameters for mean_function
                * mean_parameters = parameters for mean_function
                * enable_multi = boolean for parallel computation or not
 
        Outputs:
        ---------
            * 𝛍 = an array of size (p, 8) corresponding to the mean of the 8 classes
            of H-α plane. 
    """
 
    𝓒, mean_function, mean_parameters, enable_multi = init_parameters
    u𝓒 = np.unique(𝓒[~np.isnan(𝓒)])
    k = 0
    for class_number in u𝓒:
        𝓒[𝓒==class_number] = k
        k = k + 1
    𝛍 = wrapper_compute_all_mean_parallel(𝐗, k, 𝓒, mean_function, mean_parameters, enable_multi=enable_multi)
    return 𝛍

