import autograd.numpy as np
from multiprocessing import Process, Queue
import os
import seaborn as sns
import time
from tqdm import tqdm

from .clustering_functions import K_means

sns.set_style("darkgrid")

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def K_means_datacube(
    image,
    mask,
    features,
    window_size,
    n_classes,
    n_init,
    n_iter_max,
    eps,
    enable_multi,
    nb_threads_rows,
    nb_threads_columns
):
    """ K-means algorithm applied on an image datacube.
    It uses distances and means on locally computed features
    (e.g covariances or covariances with textures).
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
        * features = an instance of the class Feature
        * window_size = int
        * n_classes = number of classes.
        * n_init = number of initialisations of K-means
        * n_iter_max = maximum number of iterations for the K-means algorithm
        * eps = epsilon to stop K-means
        * enable_multi = enable or not parallel compuation
        * nb_threads_rows = number of threads in height
        * nb_threads_columns = number of threads to be used in column

    Outputs:
    ---------
        * C_best
    """
    if image.ndim != 3:
        raise ValueError('Error on image shape !')

    print('###################### COMPUTING FEATURES ######################')
    t_beginning = time.time()
    m_r = m_c = window_size
    n_r, n_c, p = image.shape
    X_temp = sliding_window_parallel(
        image,
        window_size,
        features.estimation,
        multi=enable_multi,
        nb_threads_rows=nb_threads_rows,
        nb_threads_columns=nb_threads_columns
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
        C, _, _, _, criterion_value = K_means(
            X,
            n_classes,
            features.distance,
            features.mean,
            init=None,
            eps=eps,
            iter_max=n_iter_max,
            enable_multi_distance=enable_multi,
            enable_multi_mean=enable_multi,
            nb_threads=os.cpu_count()
        )

        if criterion_value < best_criterion_value:
            if mask is not None:
                C_best = np.zeros((mask.shape[0], 1), dtype=np.int64) - 1
                C_best[mask] = C.reshape((C.shape[0], 1))
            else:
                C_best = C.reshape((C.shape[0], 1))
            C_best = C_best.reshape((n_r-m_r+1, n_c-m_c+1))
            C = None

    print('K-means done in %f s.' % (time.time()-t_beginning))

    return C_best


def sliding_window(
    image,
    window_size,
    function_to_compute,
    multi=False,
    queue=0,
    overlapping_window=True,
    verbose=False
):
    """ A function that allows to compute a sliding window treatment
        over a multivariate image.
        Inputs:
            * image = a numpy array of shape (n_r, n_c, p) where
                * n_r is the number of rows,
                * n_c is the number of columns
                * p is the number of canals
            * window_size = size of the squared window (3 means a 3x3 window)
            * function_to_compute = a function to compute the desired feature
            * multi = True if parallel computing
            * queue = to obtain result for parallel computation
            * overlapping_window = boolean: overlapping window or not
            * verbose = boolean
        Output:
            * result"""

    n_r, n_c, p = image.shape
    m_r = m_c = window_size
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
            local_data = image[i_r-int(m_r/2):i_r+int(m_r/2)+1,
                               i_c-int(m_c/2):i_c+int(m_c/2)+1, :]
            local_data = local_data.reshape((N, p)).T

            # Computing the function over the local data
            result_line.append(function_to_compute(local_data))

        result.append(result_line)

    if multi:
        queue.put(result)
    else:
        return result


def sliding_window_parallel(
    image,
    window_size,
    function_to_compute,
    multi=False,
    nb_threads_rows=3,
    nb_threads_columns=3,
    overlapping_window=True,
    verbose=False
):
    """ A function that is a parallelisation of sliding_window
        Inputs:
            * image = a numpy array of shape (n_r, n_c, p) where
                * n_r is the number of rows,
                * n_c is the number of columns,
                * p is the number of canals
            * window_size = size of the squared window (3 means a 3x3 window)
            * function_to_compute = a function to compute the desired quantity
            * multi = True if parallel computing, False if not
            * nb_threads_rows = number of thread to use in columns
            * nb_threads_columns = number of thread to use in columns
            * overlapping_window = boolean: overlapping window or not
            * verbose = boolean
        Outputs:
            * result.
            """

    if multi:
        # Slicing original image while taking into accound borders effects
        n_r, n_c, p = image.shape
        m_r = m_c = window_size
        image_slices_list = list()
        for i_row in range(nb_threads_rows):
            # Indexes for the sub_image for rows
            if i_row == 0:
                index_row_start = 0
            else:
                index_row_start = int(n_r/nb_threads_rows)*i_row - int(m_r/2)
            if i_row == nb_threads_rows-1:
                index_row_end = n_r
            else:
                index_row_end = int(n_r/nb_threads_rows)*(i_row+1) + int(m_r/2)

            # Slices for each row
            image_slices_list_row = list()
            for i_column in range(nb_threads_columns):
                # Indexes for the sub_image for colums
                if i_column == 0:
                    index_column_start = 0
                else:
                    index_column_start = (int(n_c/nb_threads_columns)
                                          * i_column - int(m_c/2))
                if i_column == nb_threads_columns-1:
                    index_column_end = n_c
                else:
                    index_column_end = (int(n_c/nb_threads_columns)
                                        * (i_column+1) + int(m_c/2))

                # Obtaining each slice and putting it in the list
                image_slice = image[index_row_start:index_row_end,
                                    index_column_start:index_column_end, :]
                image_slices_list_row.append(image_slice)

            # 2d list of slices
            image_slices_list.append(image_slices_list_row)

        # Freeing space
        image_slice = None
        image_slices_list_row = None

        # Serves to obtain result for each thread
        queues = [[Queue() for i_c in range(nb_threads_columns)]
                  for i_r in range(nb_threads_rows)]

        # Arguments to pass to each thread
        args = [(
            image_slices_list[i_r][i_c],
            window_size,
            function_to_compute,
            True,
            queues[i_r][i_c],
            overlapping_window
        ) for i_r in range(nb_threads_rows)
            for i_c in range(nb_threads_columns)]

        # Initialising the threads
        jobs = [Process(target=sliding_window, args=a) for a in args]

        # Starting parallel computation
        for j in jobs:
            j.start()

        # Obtaining result for each thread
        results_list = list()
        for i_r in range(nb_threads_rows):
            results_row_list = list()
            for i_c in range(nb_threads_columns):
                results_row_list.append(queues[i_r][i_c].get())
            results_list.append(results_row_list)
        results_row_list = None

        # Waiting for each thread to terminate
        if verbose:
            for j in tqdm(jobs):
                j.join()
        else:
            for j in jobs:
                j.join()

        # Now we reform the resulting image from the slices of results
        results = list()
        for i_r in range(nb_threads_rows):
            line = 0
            for line in range(len(results_list[i_r][i_c])):
                final_array_row = list()
                for i_c in range(nb_threads_columns):
                    final_array_row += results_list[i_r][i_c][line]
                results.append(final_array_row)
        final_array_row = None
    else:
        results = sliding_window(
            image,
            window_size,
            function_to_compute,
            overlapping_window=overlapping_window
        )
    return results
