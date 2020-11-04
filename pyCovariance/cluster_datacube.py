import autograd.numpy as np
from multiprocessing import Process, Queue
import os
import seaborn as sns
import sys
import time
from tqdm import tqdm
sns.set_style("darkgrid")

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import general K-means
from .clustering_functions import K_means_clustering_algorithm

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
    X_temp = sliding_window_parallel(
        image,
        window_mask,
        features.estimation,
        multi=enable_multi,
        number_of_threads_rows=number_of_threads_rows,
        number_of_threads_columns=number_of_threads_columns
    )
    X = None
    for row in range(len(X_temp)):
        for col in range(len(X_temp[row])):
            if X is None:
                X = X_temp[row][col]
            else:
                X.append(X_temp[row][col])

    image = None
    print("Done in %f s." % (time.time()-t_beginning))

    print('###################### K-MEANS CLUSTERING ######################')
    t_beginning = time.time()

    if mask is not None:
        mask = mask.reshape((-1)).astype(bool)
        X = X[mask]

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


def sliding_window(
    image,
    window_mask,
    function_to_compute,
    multi=False,
    queue=0,
    overlapping_window=True,
    verbose=False
):
    """ A function that allows to compute a sliding window treatment over a multivariate image.
        Inputs:
            * image = a numpy array of shape (n_r, n_c, p) where
                * n_r is the number of rows,
                * n_c is the number of columns
                * p is the number of canals
            * window_mask = a local mask to selection data. is a numpy boolean array.
            * function_to_compute = a function to compute the desired quantity. Must output a list.
            * multi = True if parallel computing (use the parallel function not this one), False if not
            * queue = to obtain result for parralel computation
            * overlapping_window = boolean to choose between overlapping window or not
            * verbose = boolean
        Outputs:
            * a list(list(type returned by 'function_to_compute)) corresponding to the results. First two dimensions are spatial while the third correspond to the output of function_to_compute."""

    n_r, n_c, p = image.shape
    m_r, m_c = window_mask.shape
    N = m_r*m_c
    result = list()
    if overlapping_window:
        step_rows = 1
        step_columns = 1
    else:
        step_rows = m_r
        step_columns = m_c
    iterator = range(int(m_r/2), n_r-int(m_r/2), step_rows)
    if verbose:
        iterator = tqdm(iterator)
    
    # Iterate on rows
    for i_r in iterator: 
        result_line = list()
        # Iterate on columns
        for i_c in range(int(m_c/2), n_c-int(m_c/2), step_columns):
            # Obtaining data corresponding to the neighborhood defined by the mask
            local_data = image[i_r-int(m_r/2):i_r+int(m_r/2)+1, i_c-int(m_c/2):i_c+int(m_c/2)+1, :].T.reshape((p, N))

            # Applying mask
            local_data = local_data[:, window_mask.reshape(m_r*m_c).astype(bool)]

            # Computing the function over the local data
            result_line.append(function_to_compute(local_data))

        result.append(result_line)

    if multi:
        queue.put(result)
    else:
        return result


def sliding_window_parallel(
    image,
    window_mask,
    function_to_compute,
    multi=False,
    number_of_threads_rows=3,
    number_of_threads_columns=3, 
    overlapping_window=True,
    verbose=False
):
    """ A function that is a parallelisation of sliding_window
        Inputs:
            * image = a numpy array of shape (n_r, n_c, p) where
                * n_r is the number of rows,
                * n_c is the number of columns, 
                * p is the number of canals
            * window_mask = a local mask to selection data. is a numpy boolean array.
            * function_to_compute = a function to compute the desired quantity. Must output a list.
            * multi = True if parallel computing, False if not
            * number_of_threads_rows = number of thread to use in columns 
            * number_of_threads_columns = number of thread to use in columns 
            * overlapping_window = boolean to chosse between overlapping window or not
            * verbose = boolean
        Outputs:
            * a list(list(type returned by 'function_to_compute)) corresponding to the results. First two dimensions are spatial while the third correspond to the output of function_to_compute.
            """

    if multi:
        # Slicing original image while taking into accound borders effects
        n_r, n_c, p = image.shape
        m_r, m_c = window_mask.shape
        image_slices_list = list() # Will contain each slice
        for i_row in range(number_of_threads_rows):
            # Indexes for the sub_image for rows
            if i_row == 0:
                index_row_start = 0
            else:
                index_row_start = int(n_r/number_of_threads_rows)*i_row - int(m_r/2)
            if i_row == number_of_threads_rows-1:
                index_row_end = n_r
            else:
                index_row_end = int(n_r/number_of_threads_rows)*(i_row+1) + int(m_r/2)

            # Slices for each row
            image_slices_list_row = list()
            for i_column in range(number_of_threads_columns):
                # Indexes for the sub_image for colums
                if i_column == 0:
                    index_column_start = 0
                else:
                    index_column_start = int(n_c/number_of_threads_columns)*i_column - int(m_c/2)
                if i_column == number_of_threads_columns-1:
                    index_column_end = n_c
                else:
                    index_column_end = int(n_c/number_of_threads_columns)*(i_column+1) + int(m_c/2)

                # Obtaining each slice and putting it in the list
                image_slice = image[index_row_start:index_row_end, index_column_start:index_column_end, :]
                image_slices_list_row.append(image_slice)

            # 2d list of slices
            image_slices_list.append(image_slices_list_row)

        # Freeing space
        image_slice = None
        image_slices_list_row = None

        # Serves to obtain result for each thread
        queues = [[Queue() for i_c in range(number_of_threads_columns)] for i_r in range(number_of_threads_rows)]

        # Arguments to pass to each thread
        args = [(
            image_slices_list[i_r][i_c],
            window_mask,
            function_to_compute,
            True,
            queues[i_r][i_c],
            overlapping_window
        ) for i_r in range(number_of_threads_rows) for i_c in range(number_of_threads_columns)] 

        # Initialising the threads
        jobs = [Process(target=sliding_window, args=a) for a in args]

        # Starting parallel computation
        for j in jobs: j.start()

        # Obtaining result for each thread
        results_list = list() # Results container
        for i_r in range(number_of_threads_rows):
            results_row_list = list()
            for i_c in range(number_of_threads_columns):
                results_row_list.append( queues[i_r][i_c].get() )
            results_list.append(results_row_list)
        results_row_list = None

        # Waiting for each thread to terminate
        if verbose:
            for j in tqdm(jobs): j.join()
        else:
            for j in jobs: j.join()

        # Now we reform the resulting image from the slices of results
        results = list()
        for i_r in range(number_of_threads_rows):
            line = 0
            for line in range(len(results_list[i_r][i_c])):
                final_array_row = list()
                for i_c in range(number_of_threads_columns):
                    final_array_row += results_list[i_r][i_c][line]
                results.append(final_array_row)
        final_array_row = None
    else:
        results = sliding_window(image, window_mask, function_to_compute, overlapping_window=overlapping_window)
    return results
