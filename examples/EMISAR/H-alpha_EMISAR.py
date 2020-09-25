import autograd.numpy as np
import os
import sys
import time

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from pyCovariance.H_alpha_functions import cluster_image_by_H_alpha
from pyCovariance.utils import enable_latex_infigures, plot_segmentation, save_figure

# DEBUG mode for faster debugging
DEBUG = True
if DEBUG:
    print('DEBUG mode enabled !!!')
    print()
    SIZE_CROP = 100

# Activate latex in figures (or not)
LATEX_IN_FIGURES = False
if LATEX_IN_FIGURES:
  enable_latex_infigures()

# Enable parallel processing (or not)
ENABLE_MULTI = True
NUMBER_OF_THREADS_ROWS = 4
NUMBER_OF_THREADS_COLUMNS = 2
if NUMBER_OF_THREADS_ROWS*NUMBER_OF_THREADS_COLUMNS != os.cpu_count():
    print('ERROR: all cpus are not used ...')
    sys.exit(1)

# Dataset
PATH = 'data/EMISAR/EMISAR_data.npy'
RESOLUTION = [0.749, 1.499] # resolution in meters

# Window size to compute features
WINDOWS_SHAPE = (7,7)
windows_mask = np.ones(WINDOWS_SHAPE)

print('################################################')
print('Reading dataset') 
print('################################################')
t_beginning = time.time()
image = np.load(PATH)
image = image[:,3:,:] # Removing border which is on the left
if DEBUG:
    center = np.array(image.shape[0:2])//2
    half_height = SIZE_CROP//2
    half_width = SIZE_CROP//2
    image = image[center[0]-half_height:center[0]+half_height, center[1]-half_width:center[1]+half_width]
print("Done in %f s." % (time.time()-t_beginning))
print()

print('################################################')
print('H-alpha decomposition') 
print('################################################')
C = cluster_image_by_H_alpha(
    image,
    windows_mask,
    multi=ENABLE_MULTI,
    number_of_threads_rows=NUMBER_OF_THREADS_ROWS,
    number_of_threads_columns=NUMBER_OF_THREADS_COLUMNS
)

# Plotting
plot_segmentation(C, aspect=RESOLUTION[0]/RESOLUTION[1])
save_figure('figures', 'fig_H_alpha_EMISAR')
