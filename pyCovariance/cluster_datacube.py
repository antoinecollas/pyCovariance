import autograd.numpy as np
import os
import seaborn as sns
import sys
import time
from tqdm import tqdm
sns.set_style("darkgrid")

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import general K-means
from pyCovariance.clustering_functions import K_means_clustering_algorithm

# import sliding window function
from pyCovariance.sliding_window import sliding_window_parallel

def K_means_datacube(
    image,
    mask,
    features,
    windows_shape,
    n_classes,
    n_init,
    n_iter_max,
    eps,
    enable_multi,
    number_of_threads_rows,
    number_of_threads_columns
):
    """ K-means algorithm applied on an image datacube. It uses distances and means on locally computed features (e.g covariances or covariances with textures).
    --------------------------------------------------------------
    Inputs:
    --------
        * images = (h, w, p) numpy array with:
            * h = height of the image
            * w = width of the image
            * p = size of each pixel
        * mask = (h, w) numpy array to select pixels to cluster:
            * h = height of the image
            * w = width of the image
        * features = an instance of the class BaseClassFeatures from pyCovariance.features
        * windows_shape = (h, w) tuple:
            * h: height of the window in pixels
            * w: width of the window in pixels
        * n_classes = number of classes.
        * n_init = number of initialisations of K-means in the case where centers of classes are randomly chosen.
        * n_iter_max = maximum number of iterations for the K-means algorithm
        * eps = epsilon to stop K-means
        * enable_multi = enable or not parallel compuation
        * number_of_threads_rows = number of threads to be used to cut the image in height
        * number_of_threads_columns = number of threads to be used to cut the image in column

    Outputs:
    ---------
        * C_its
    """
    if len(image.shape) != 3:
        raise ValueError('Error on image shape !')
    
    C_init = None

    print('###################### COMPUTING FEATURES ######################')
    t_beginning = time.time()
    window_mask = np.ones(windows_shape)
    m_r, m_c = window_mask.shape
    N = m_r*m_c
    n_r, n_c, p = image.shape
    features_temp = sliding_window_parallel(
        image,
        window_mask,
        features.estimation,
        multi=enable_multi,
        number_of_threads_rows=number_of_threads_rows,
        number_of_threads_columns=number_of_threads_columns
    )
    X = [i for row in features_temp for i in row]
    image = None
    print("Done in %f s." % (time.time()-t_beginning))

    print('###################### K-MEANS CLUSTERING ######################')
    t_beginning = time.time()

    if mask is not None:
        mask = mask.reshape((-1))
        X_new = list()
        for x, m in zip(X, mask):
            if m.astype(bool):
                X_new.append(x)
        X = X_new

    best_criterion_value = np.inf
    for _ in tqdm(range(n_init)):
        C, _, _, _, criterion_value = K_means_clustering_algorithm(
            X,
            n_classes,
            features.distance,
            features.mean,
            init=None,
            eps=eps,
            iter_max = n_iter_max,
            enable_multi_distance=enable_multi,
            enable_multi_mean=enable_multi,
            number_of_threads=os.cpu_count()
        )

        if criterion_value < best_criterion_value:
            if mask is not None:
                C_its = np.zeros((mask.shape[0], 1)) - 1
                C_its[mask] = C.reshape((C.shape[0], 1))
            else:
                C_its = C.reshape((C.shape[0], 1))
            C_its += 1
            C_its = C_its.reshape((n_r-m_r+1, n_c-m_c+1, 1))
            C = None

    print('K-means done in %f s.' % (time.time()-t_beginning))
 
    return C_its
