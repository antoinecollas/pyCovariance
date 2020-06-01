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

# import functions related to covariance features
from clustering_SAR.covariance_clustering_functions import vech_SCM, Riemannian_distance_covariance, Riemannian_mean_covariance

# import functions related to covariance and texture features
from clustering_SAR.covariance_and_texture_clustering_functions import compute_feature_Covariance_texture, Riemannian_distance_covariance_texture, Riemannian_mean_covariance_texture

# import functions related to H-alpha decomposition
from clustering_SAR.H_alpha_functions import cluster_image_by_H_alpha, initialise_H_alpha

# import sliding window function
from clustering_SAR.multivariate_images_tools import sliding_windows_treatment_image_time_series_parallel

def K_means_SAR_datacube(
    images,
    features,
    windows_shape,
    k_means_nb_iter_max,
    enable_multi,
    number_of_threads_rows,
    number_of_threads_columns
):
    """ K-means algorithm applied on a SAR datacube. It uses Riemannian distances and means on locally computed features (covariances or covariances with textures).
    ---------------------------------------------------------------
        Inputs:
        --------
            * images = (H, W, p, T) numpy array with:
                * H = height of the image
                * W = width of the image
                * p = size of each pixel
                * T = number of images (size of the time series)
            * features = string corresponding to the features to compute.
            * windows_shape = (h, w) tuple:
                * h: height of the window in pixels
                * w: width of the window in pixels
            * K_means_nb_iter_max = maximum number of iterations for the K-means algorithm
            * enable_multi = enable or not parallel compuation
            * number_of_threads_rows = number of threads to be used to cut the image in height
            * number_of_threads_columns = number of threads to be used to cut the image in column

        Outputs:
        ---------
            * C_its
    """
    
    assert (features == 'covariance') or (features == 'covariance_texture')

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
    if features == 'covariance':
        function_to_compute = vech_SCM
    elif features == 'covariance_texture':
        function_to_compute = compute_feature_Covariance_texture
    function_args = (0.01, 20)
    for t in range(T):
        print("Treating image %d of %d" %(t+1,T))
        feature_temp = sliding_windows_treatment_image_time_series_parallel(
            images[:,:,:,t].reshape(n_r,n_c,p,1),
            windows_mask,
            function_to_compute,
            function_args,
            multi=enable_multi,
            number_of_threads_rows=number_of_threads_rows,
            number_of_threads_columns=number_of_threads_columns
        )
        images = images[:,:,:,1:] # Freeing memory space
        print(((n_r-m_r+1)*(n_c-m_c+1), int(p*(p+1)/2)+N))
        if features == 'covariance':
            X.append(feature_temp.reshape(((n_r-m_r+1)*(n_c-m_c+1), int(p*(p+1)/2))).T)
        elif features == 'covariance_texture':
            X.append(feature_temp.reshape(((n_r-m_r+1)*(n_c-m_c+1), int(p*(p+1)/2)+N)).T)
        feature_temp = None # Freeing memory space
    X = np.hstack(X)
    images = None
    print("Done in %f s." % (time.time()-t_beginning))
    print()

    print('################################################')
    print('K-means clustering') 
    print('################################################')
    mean_parameters = [1.0, 0.95, 1e-9, 5, False, 0]
    if features == 'covariance':
        distance = Riemannian_distance_covariance
        mean = Riemannian_mean_covariance
    elif features == 'covariance_texture':
        distance = Riemannian_distance_covariance_texture
        mean = Riemannian_mean_covariance_texture
        mean_parameters =  [p, N] + mean_parameters
    t_beginning = time.time()
    K = len(np.unique(C[~np.isnan(C)]))
    C, mu, i, delta = K_means_clustering_algorithm(
        X,
        K,
        distance,
        (p, N), 
        mean,
        mean_parameters,
        init=initialise_H_alpha,
        eps=1e-6,
        init_parameters=[C, Riemannian_mean_covariance_texture, [p, N, 1.0, 0.95, 1e-9, 5, False, 0], True],
        iter_max = k_means_nb_iter_max,
        enable_multi=enable_multi,
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
