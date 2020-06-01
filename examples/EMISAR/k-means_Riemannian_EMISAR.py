import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys
import time

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.cluster_datacube import K_means_SAR_datacube
from clustering_SAR.generic_functions import enable_latex_infigures, plot_Pauli_SAR, save_figure

# DEBUG mode for faster debugging
DEBUG = True
if DEBUG:
    print('DEBUG mode enabled !!!')
    print()
    SIZE_CROP = 100

# Activate latex in figures (or not)
LATEX_IN_FIGURES = True
if LATEX_IN_FIGURES:
  enable_latex_infigures()

# Enable parallel processing (or not)
ENABLE_MULTI = True
NUMBER_OF_THREADS_ROWS = 4
NUMBER_OF_THREADS_COLUMNS = 2
if NUMBER_OF_THREADS_ROWS*NUMBER_OF_THREADS_COLUMNS != os.cpu_count():
    print('ERROR: all cpus are not used ...')
    sys.exit(1)
NUMBER_OF_THREADS = os.cpu_count() 

# Dataset
PATH = 'data/EMISAR/EMISAR_data.npy'
RESOLUTION = [0.749, 1.499] # resolution in meters
T = 1

# Window size to compute features
WINDOWS_SHAPE = (7,7)

# features used to cluster the its
FEATURES = 'covariance'

# K-means parameter
if DEBUG:
    K_MEANS_NB_ITER_MAX = 2
else:
    K_MEANS_NB_ITER_MAX = 10

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
plot_Pauli_SAR(image, RESOLUTION)
n_r, n_c, p = image.shape
image = image.reshape(n_r, n_c, p, T)
print("Done in %f s." % (time.time()-t_beginning))
print()

C_its = K_means_SAR_datacube(
    image,
    FEATURES,
    WINDOWS_SHAPE,
    K_MEANS_NB_ITER_MAX,
    ENABLE_MULTI,
    NUMBER_OF_THREADS_ROWS,
    NUMBER_OF_THREADS_COLUMNS
)

# Plotting
cmap = plt.cm.inferno_r
# define the bins and normalize
K = len(np.unique(C_its))
bounds = np.linspace(0, K, K+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
for t in range(T):
    fig = plt.figure(figsize=(16,9), dpi=80, facecolor='w')
    plt.imshow(C_its[:,:,t], aspect=RESOLUTION[0]/RESOLUTION[1], cmap=cmap, norm=norm)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()

save_figure('figures', 'fig_EMISAR')
