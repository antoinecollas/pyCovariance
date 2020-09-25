import autograd.numpy as np
from multiprocessing import Process, Queue
from tqdm import tqdm


def sliding_windows_treatment_image_time_series(
    image,
    windows_mask,
    function_to_compute,
    multi=False,
    queue=0,
    overlapping_windows=True,
    verbose=False
):
    """ A function that allowing to compute a sliding windows treatment over a multivariate
        image time series.
        Inputs:
            * image = a numpy array of shape (n_r,n_c,p,T) where n_r is the number of rows,
                      n_c is the number of columns, p is the number of canals and T is the length
                      of the time series.
            * windows_mask = a local mask to selection data. is a numpy boolean array.
            * function_to_compute = a function to compute the desired quantity. Must output a list.
            * multi = True if parallel computing (use the parallel function not this one), False if not
            * queue = to obtain result for parralel computation
            * overlapping_windows = boolean to chosse between overlapping windows or not
            * verbose = boolean
        Outputs:
            * a 3-d array corresponding to the results. First two dimensions are spatial while the third correspond
              to the output of function_to_compute."""

    n_r, n_c, p, T = image.shape
    m_r, m_c = windows_mask.shape
    N = m_r*m_c
    result = []
    if overlapping_windows:
        step_rows = 1
        step_columns = 1
    else:
        step_rows = m_r
        step_columns = m_c
    iterator = range(int(m_r/2),n_r-int(m_r/2),step_rows)
    if verbose:
        iterator = tqdm(iterator)
    for i_r in iterator: # Iterate on rows
        result_line = []
        for i_c in range(int(m_c/2),n_c-int(m_c/2),step_columns): # Iterate on columns

            # Obtaining data corresponding to the neighborhood defined by the mask
            local_data = image[i_r-int(m_r/2):i_r+int(m_r/2)+1, i_c-int(m_c/2):i_c+int(m_c/2)+1, :, 0].T.reshape((p,N))
            for t in range(1,T):
                local_data = np.dstack((local_data, image[i_r-int(m_r/2):i_r+int(m_r/2)+1, i_c-int(m_c/2):i_c+int(m_c/2)+1, :, t].T.reshape((p,N))))
            
            # Applying mask
            local_data = local_data.reshape((p,N,T))
            local_data = local_data[:,windows_mask.reshape(m_r*m_c).astype(bool),:]

            # Computing the function over the local data
            result_line.append(function_to_compute(local_data))
        result.append(result_line)
        
    if multi:
        queue.put(result)
    else:
        return np.array(result)



def sliding_windows_treatment_image_time_series_parallel(
    image,
    windows_mask,
    function_to_compute,
    multi=False,
    number_of_threads_rows=3,
    number_of_threads_columns=3, 
    overlapping_windows=True,
    verbose=False
):
    """ A function that is a prallelisation of sliding_windows_treatment_image_time_series
        Inputs:
            * image = a numpy array of shape (n_r,n_c,p,T) where n_r is the number of rows,
                      n_c is the number of columns, p is the number of canals and T is the length
                      of the time series.
            * windows_mask = a local mask to selection data. is a numpy boolean array.
            * function_to_compute = a function to compute the desired quantity. Must output a list.
            * multi = True if parallel computing, False if not
            * number_of_threads_columns = number of thread to use in columns 
                (total threads = number of cores of the machine in general)
            * number_of_threads_rows = number of thread to use in columns 
                (total threads = number of cores of the machine in general) 
            * overlapping_windows = boolean to chosse between overlapping windows or not
            * verbose = boolean
        Outputs:
            * number_of_threads_columns = number of thread to use in columns 
                (total threads = number of cores of the machine in general)"""

    if multi:
        # Slicing original image while taking into accound borders effects
        n_r, n_c, p, T = image.shape
        m_r, m_c = windows_mask.shape
        image_slices_list = [] # Will contain each slice
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
            image_slices_list_row = []
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
                image_slice = image[index_row_start:index_row_end, index_column_start:index_column_end, :, :]
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
            windows_mask,
            function_to_compute,
            True,
            queues[i_r][i_c],
            overlapping_windows
        ) for i_r in range(number_of_threads_rows) for i_c in range(number_of_threads_columns)] 

        # Initialising the threads
        jobs = [Process(target=sliding_windows_treatment_image_time_series, args=a) for a in args]

        # Starting parallel computation
        for j in jobs: j.start()

        # Obtaining result for each thread
        results_list = [] # Results container
        for i_r in range(number_of_threads_rows):
            results_row_list = []
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
        results = []
        for i_r in range(number_of_threads_rows):
            final_array_row = []
            for i_c in range(number_of_threads_columns):
                final_array_row.append(results_list[i_r][i_c])
            results.append(np.hstack(final_array_row))
        
        results = np.vstack(results)
        final_array_row = None

    else:
        results = sliding_windows_treatment_image_time_series(image, windows_mask, 
                                        function_to_compute, overlapping_windows=overlapping_windows)
    return results
   
    
