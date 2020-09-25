import autograd.numpy as np
import warnings

from .clustering_functions import wrapper_compute_all_mean_parallel
from .covariance_clustering_functions import vech_SCM
from .utils import * 
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

    return c


def compute_h_alpha_class(X):
    """ Compute H-α values given data in input and assign to a class
    The computation of H and Alpha follows 'Unsupervised Classification Using Polarimetric Decomposition and the Complex Wishart Classifier' from Lee et al.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation of covariance matrix

        Outputs:
        ---------
            * the class corresponding to a zone in the H-α plane
    """
    # Pauli transformation
    X[1, :] = np.sqrt(2) * X[1, :]
    cov = SCM(np.squeeze(X))
    N = (1/np.sqrt(2)) * np.array([[1, 0, 1], [1, 0, -1], [0, np.sqrt(2), 0]])
    T = N @ cov @ N.T

    eigvalues, eigvectors = np.linalg.eigh(T)
    eigvalues = eigvalues/np.sum(eigvalues)

    α_vector = np.degrees(np.arccos(np.abs(eigvectors[0,:])))

    H = - np.dot(eigvalues, np.log(eigvalues)/np.log(3))
    α = np.dot(eigvalues, α_vector)

    return [assign_class_H_α(H, α)]


def cluster_image_by_H_alpha(
    image,
    windows_mask,
    multi=False, 
    number_of_threads_rows=4,
    number_of_threads_columns=4
):
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
    𝓒 = sliding_windows_treatment_image_time_series_parallel(
        image.reshape(n_r,n_c,p,1), 
        windows_mask,
        compute_h_alpha_class, 
        multi=multi, 
        number_of_threads_rows=number_of_threads_rows,
        number_of_threads_columns=number_of_threads_columns
    )

    return 𝓒.reshape(n_r-m_r+1, n_c-m_c+1)
