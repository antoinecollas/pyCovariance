import autograd.numpy as np
from multiprocessing import Process, Queue
import os
import time
from tqdm import tqdm

from .clustering import _K_means


def K_means_datacube(
    image,
    mask,
    feature,
    window_size,
    n_classes,
    n_init,
    n_max_iter,
    tol,
    n_jobs_rows,
    n_jobs_columns,
    verbose=True
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
        * feature = a Feature from pyCovariance.features
            e.g see pyCovariance/features/covariance.py
        * window_size = int
        * n_classes = number of classes.
        * n_init = number of initialisations of K-means
        * n_max_iter = maximum number of iterations for the K-means algorithm
        * tol = tolilon to stop K-means
        * n_jobs_rows = number of threads in height
        * n_jobs_columns = number of threads to be used in column
        * verbose = bool

    Outputs:
    ---------
        * C_best
        * all_criterion_values = list of within classes variances
                                of all K_means done
    """
    if image.ndim != 3:
        raise ValueError('Error on image shape !')

    if mask is not None:
        h = w = window_size//2
        mask = mask[h:-h, w:-w]

    if verbose:
        print('#################### COMPUTING FEATURES ####################')
    t_beginning = time.time()
    m_r = m_c = window_size
    n_r, n_c, p = image.shape
    p = image.shape[2]
    N = window_size**2
    feature = feature(p, N)
    X_temp = sliding_window_parallel(
        image,
        window_size,
        feature.estimation,
        n_jobs_rows=n_jobs_rows,
        n_jobs_columns=n_jobs_columns
    )
    X = None
    for row in range(len(X_temp)):
        for col in range(len(X_temp[row])):
            if X is None:
                X = X_temp[row][col]
            else:
                X.append(X_temp[row][col])

    image = None
    if verbose:
        print("Done in %f s." % (time.time()-t_beginning))
        print('##################### K-MEANS CLUSTERING #####################')
    t_beginning = time.time()

    if mask is not None:
        mask = mask.reshape((-1)).astype(bool)
        X = X[mask]

    C, _, _, _, all_criterion_values = _K_means(
        X,
        n_classes,
        feature.distance,
        feature.mean,
        init=None,
        tol=tol,
        n_init=n_init,
        max_iter=n_max_iter,
        n_jobs=n_jobs_rows*n_jobs_columns,
        verbose=verbose
    )

    if mask is not None:
        C_best = np.zeros((mask.shape[0], 1), dtype=np.int64) - 1
        C_best[mask] = C.reshape((C.shape[0], 1))
    else:
        C_best = C.reshape((C.shape[0], 1))
    C_best = C_best.reshape((n_r-m_r+1, n_c-m_c+1))

    if verbose:
        print('K-means done in %f s.' % (time.time()-t_beginning))

    return C_best, all_criterion_values


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
    n_jobs_rows=1,
    n_jobs_columns=1,
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
            * n_jobs_rows = number of thread to use in columns
            * n_jobs_columns = number of thread to use in columns
            * overlapping_window = boolean: overlapping window or not
            * verbose = boolean
        Outputs:
            * result.
            """

    if (n_jobs_rows > 1) or (n_jobs_columns > 1):
        # Slicing original image while taking into accound borders effects
        n_r, n_c, p = image.shape
        m_r = m_c = window_size
        image_slices_list = list()
        for i_row in range(n_jobs_rows):
            # Indexes for the sub_image for rows
            if i_row == 0:
                index_row_start = 0
            else:
                index_row_start = int(n_r/n_jobs_rows)*i_row - int(m_r/2)
            if i_row == n_jobs_rows-1:
                index_row_end = n_r
            else:
                index_row_end = int(n_r/n_jobs_rows)*(i_row+1) + int(m_r/2)

            # Slices for each row
            image_slices_list_row = list()
            for i_column in range(n_jobs_columns):
                # Indexes for the sub_image for colums
                if i_column == 0:
                    index_column_start = 0
                else:
                    index_column_start = (int(n_c/n_jobs_columns)
                                          * i_column - int(m_c/2))
                if i_column == n_jobs_columns-1:
                    index_column_end = n_c
                else:
                    index_column_end = (int(n_c/n_jobs_columns)
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
        queues = [[Queue() for i_c in range(n_jobs_columns)]
                  for i_r in range(n_jobs_rows)]

        # Arguments to pass to each thread
        args = [(
            image_slices_list[i_r][i_c],
            window_size,
            function_to_compute,
            True,
            queues[i_r][i_c],
            overlapping_window
        ) for i_r in range(n_jobs_rows)
            for i_c in range(n_jobs_columns)]

        # Initialising the threads
        jobs = [Process(target=sliding_window, args=a) for a in args]

        # Starting parallel computation
        for j in jobs:
            j.start()

        # Obtaining result for each thread
        results_list = list()
        for i_r in range(n_jobs_rows):
            results_row_list = list()
            for i_c in range(n_jobs_columns):
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
        for i_r in range(n_jobs_rows):
            line = 0
            for line in range(len(results_list[i_r][i_c])):
                final_array_row = list()
                for i_c in range(n_jobs_columns):
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
