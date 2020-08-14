import numpy as np
import os
import seaborn as sns
import sys
import time
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
    images,
    features,
    windows_shape,
    init,
    k_means_nb_iter_max,
    enable_multi,
    number_of_threads_rows,
    number_of_threads_columns
):
    """ K-means algorithm applied on a image time series datacube. It uses distances and means on locally computed features (e.g covariances or covariances with textures).
    --------------------------------------------------------------
    Inputs:
    --------
        * images = (H, W, p, T) numpy array with:
            * H = height of the image
            * W = width of the image
            * p = size of each pixel
            * T = number of images (size of the time series)
        * features = an instance of the class BaseClassFeatures from clustering_SAR.features
        * windows_shape = (h, w) tuple:
            * h: height of the window in pixels
            * w: width of the window in pixels
        * init = initialisation of K-means. Either an integer or 'H-alpha'. An integer corresponds to a number of classes. In this case, center of classes are randomly chosen.
        * K_means_nb_iter_max = maximum number of iterations for the K-means algorithm
        * enable_multi = enable or not parallel compuation
        * number_of_threads_rows = number of threads to be used to cut the image in height
        * number_of_threads_columns = number of threads to be used to cut the image in column

    Outputs:
    ---------
    * C_its
    """
    if len(images.shape) == 3:
        images = images.reshape(*images.shape, 1)
    
    assert (type(init) is int) or init=='H-alpha', 'Error initialisation in K-means arguments'
    C = None
    if init=='H-alpha':
        print('################################################')
        print('Initialisation: H-alpha')
        print('################################################')
        windows_mask = np.ones(windows_shape)
        n_r, n_c, p, T = images.shape
        m_r, m_c = windows_mask.shape
        N = m_r*m_c
        C = []
        for t in range(T):
            print("Treating image %d of %d" %(t+1,T))
            C_tmp = cluster_image_by_H_alpha(
                images[:,:,:,t],
                windows_mask,
                multi=enable_multi,
                number_of_threads_rows=number_of_threads_rows,
                number_of_threads_columns=number_of_threads_columns
            )
            C.append(C_tmp.reshape(C_tmp.size, 1).T)
        C = np.squeeze(np.hstack(C))
        C_tmp = None
        print()

    print('################################################')
    print('Computing features')
    print('################################################')
    t_beginning = time.time()
    windows_mask = np.ones(windows_shape)
    m_r, m_c = windows_mask.shape
    N = m_r*m_c
    n_r, n_c, p, T = images.shape
    X = []
    for t in range(T):
        print("Treating image %d of %d" %(t+1,T))
        feature_temp = sliding_windows_treatment_image_time_series_parallel(
            images[:,:,:,t].reshape(n_r,n_c,p,1),
            windows_mask,
            features.estimation,
            multi=enable_multi,
            number_of_threads_rows=number_of_threads_rows,
            number_of_threads_columns=number_of_threads_columns
        )
        images = images[:,:,:,1:] # Freeing memory space
        X.append(feature_temp.reshape(((n_r-m_r+1)*(n_c-m_c+1), -1)).T)
        feature_temp = None # Freeing memory space
    X = np.hstack(X)
    images = None
    print("Done in %f s." % (time.time()-t_beginning))
    print()

    print('################################################')
    print('K-means clustering') 
    print('################################################')
    t_beginning = time.time()
    if type(init) is int:
        K = init
    else:
        K = len(np.unique(C))
    C, mu, i, delta = K_means_clustering_algorithm(
        X,
        K,
        features.distance,
        features.mean,
        init=C,
        eps=1e-6,
        iter_max = k_means_nb_iter_max,
        enable_multi_distance=enable_multi,
        enable_multi_mean=enable_multi,
        number_of_threads=os.cpu_count()
    )
    X = None
    print("K-means done in %f s." % (time.time()-t_beginning))
    print()

    # Reformatting ITS in the form of images
    C_its = np.empty(((n_r-m_r+1), (n_c-m_c+1), T))
    for t in range(T):
        C_its[:,:,t] = C[t*(n_r-m_r+1)*(n_c-m_c+1):(t+1)*(n_r-m_r+1)*(n_c-m_c+1)].reshape((n_r-m_r+1,n_c-m_c+1))
    C = None
 
    return C_its
