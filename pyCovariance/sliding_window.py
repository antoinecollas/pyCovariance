import autograd.numpy as np
from multiprocessing import Process, Queue
from tqdm import tqdm


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
