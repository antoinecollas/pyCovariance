import numpy as np
import os
import seaborn as sns
import sys
import time
from tqdm import tqdm
sns.set_style("darkgrid")

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

# import general K-means
from clustering_SAR.clustering_functions import K_means_clustering_algorithm

# import functions related to H-alpha decomposition
from clustering_SAR.H_alpha_functions import cluster_image_by_H_alpha

# import sliding window function
from clustering_SAR.multivariate_images_tools import sliding_windows_treatment_image_time_series_parallel

def K_means_datacube(
    image,
    mask,
    features,
    windows_shape,
    init,
    n_init,
    k_means_nb_iter_max,
    eps,
    enable_multi,
    number_of_threads_rows,
    number_of_threads_columns
):
    """ K-means algorithm applied on an image datacube. It uses distances and means on locally computed features (e.g covariances or covariances with textures).
    --------------------------------------------------------------
    Inputs:
    --------
        * images = (H, W, p) numpy array with:
            * H = height of the image
            * W = width of the image
            * p = size of each pixel
        * mask = (H, W, p) numpy array to select pixels to cluster:
            * H = height of the image
            * W = width of the image
            * p = size of each pixel
        * features = an instance of the class BaseClassFeatures from clustering_SAR.features
        * windows_shape = (h, w) tuple:
            * h: height of the window in pixels
            * w: width of the window in pixels
        * init = initialisation of K-means. Either an integer or 'H-alpha'. An integer corresponds to a number of classes. In this case, centers of classes are randomly chosen.
        * n_init = number of initialisations of K-means in the case where centers of classes are randomly chosen.
        * K_means_nb_iter_max = maximum number of iterations for the K-means algorithm
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
    
    assert (type(init) is int and n_init>0) or (init=='H-alpha' and n_init==1), 'Error initialisation in K-means arguments'
    C_init = None
    if init=='H-alpha':
        print('###################### INITIALISATION: H-ALPHA ######################')
        windows_mask = np.ones(windows_shape)
        n_r, n_c, p = image.shape
        m_r, m_c = windows_mask.shape
        N = m_r*m_c
        C_init = cluster_image_by_H_alpha(
            image,
            windows_mask,
            multi=enable_multi,
            number_of_threads_rows=number_of_threads_rows,
            number_of_threads_columns=number_of_threads_columns
        )
        print()

    print('###################### COMPUTING FEATURES ######################')
    t_beginning = time.time()
    windows_mask = np.ones(windows_shape)
    m_r, m_c = windows_mask.shape
    N = m_r*m_c
    n_r, n_c, p = image.shape
    feature_temp = sliding_windows_treatment_image_time_series_parallel(
        image.reshape((n_r, n_c, p, 1)),
        windows_mask,
        features.estimation,
        multi=enable_multi,
        number_of_threads_rows=number_of_threads_rows,
        number_of_threads_columns=number_of_threads_columns
    )
    X = feature_temp.reshape(((n_r-m_r+1)*(n_c-m_c+1), -1)).T
    feature_temp = None # Freeing memory space
    image = None
    print("Done in %f s." % (time.time()-t_beginning))

    print('###################### K-MEANS CLUSTERING ######################')
    t_beginning = time.time()
   
    if mask is not None:
        mask = mask.reshape((-1))
        X = X[:, mask]

    best_criterion_value = np.inf
    for _ in tqdm(range(n_init)):
        if type(init) is int:
            K = init
        else:
            K = len(np.unique(C_init))
        C, _, _, _, criterion_value = K_means_clustering_algorithm(
            X,
            K,
            features.distance,
            features.mean,
            init=C_init,
            eps=eps,
            iter_max = k_means_nb_iter_max,
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
