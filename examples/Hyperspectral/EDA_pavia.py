import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat
import scipy.stats as stats
import seaborn as sns
from sklearn.decomposition import PCA
import statsmodels.api as sm
import sys

sns.set_style("darkgrid")

#######################################################
# BEGINNING OF HYPERPARAMETERS
#######################################################

# Dataset
PATH = 'data/Pavia/PaviaU.mat'
KEY_DICT_PAVIA = 'paviaU'
NB_BANDS_TO_SELECT = 2
RESOLUTION = [1.3, 1.3] # resolution in meters

# ground truth path
PATH_GT = 'data/Pavia/PaviaU_gt.mat'
gt = loadmat(PATH_GT)['paviaU_gt']

# path of directory to dave results
DIRECTORY_RESULTS = 'results/EDA/'

#######################################################
# END OF HYPERPARAMETERS
#######################################################

if not os.path.exists(DIRECTORY_RESULTS):
    os.makedirs(DIRECTORY_RESULTS)

# load image and ground truth
image = loadmat(PATH)[KEY_DICT_PAVIA]
gt = loadmat(PATH_GT)['paviaU_gt']

# center pixels
h, w, p = image.shape
image = image.reshape((-1, image.shape[-1]))
mean = np.mean(image, axis=0)
image = image - mean
# check pixels are centered
assert (np.abs(np.mean(image, axis=0)) < 1e-9).all()

# apply PCA
pca = PCA()
image = pca.fit_transform(image)
# check pixels are still centered
assert (np.abs(np.mean(image, axis=0)) < 1e-9).all()
# reshape image
image = image.reshape((h, w, p))
image = image[:, :, :NB_BANDS_TO_SELECT]

temp = image.reshape((-1, NB_BANDS_TO_SELECT))
print('Global mean=', np.mean(temp, axis=0))
print('Global SCM=', temp.T@temp/temp.shape[0])
print()

classes = np.unique(gt)[1:]
for i in classes:
    print('Class:', i)
    pixels = np.copy(image[gt==i])
    mean = np.mean(pixels, axis=0)
    print('mean=', mean)
    pixels = pixels - mean
    SCM = (pixels.T @ pixels) / pixels.shape[0]
    print('SCM=', SCM)
    print()

    d = np.diagonal(pixels@np.linalg.inv(SCM)@pixels.T)

    fig = plt.figure(1, figsize=(10,5))
    ax = plt.subplot(121)
    
    probplot = sm.ProbPlot(d, stats.chi2, distargs=(NB_BANDS_TO_SELECT,))
    fig = probplot.ppplot(line='45', ax=ax)
    plt.title('Probability-Probability plot')
    
    plt.subplot(122)
    
    d_full = np.sort(d)
    d = d_full[:int(len(d_full)*0.9)]
    x = np.linspace(np.min(d), np.max(d), num=1000)
    pdf = stats.chi2.pdf(x, df=NB_BANDS_TO_SELECT)
    plt.hist(d, bins='auto', density=True)
    plt.plot(x, pdf)
    plt.title('Histogram vs chi2 pdf. (90% des données)')
    
    fig.savefig(os.path.join(DIRECTORY_RESULTS, 'class_'+str(i)+'.png'))
    
    plt.clf()
