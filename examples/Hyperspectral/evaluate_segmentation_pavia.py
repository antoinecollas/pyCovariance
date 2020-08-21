import copy
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import sys

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.evaluation import assign_classes_segmentation_to_gt, compute_mIoU, plot_segmentation, plot_TP_FP_FN_segmentation

#######################################################
#######################################################
# BEGINNING OF HYPERPARAMETERS
#######################################################
#######################################################

# segmentation path (by default: get last folder)
FOLDER_REGEX = 'results/*'
folders_result = glob.glob(FOLDER_REGEX)
folders_result.sort()
folder_result = folders_result[-1]
print('Folder used to get segmentations:', folder_result)
segmentations_paths = glob.glob(folder_result+'/*.npy')
segmentations_paths.sort()

if len(segmentations_paths) == 0:
    print('No npy files found ...')
    sys.exit(1)

# ground truth path
PATH_GT = 'data/Pavia/PaviaU_gt.mat'
gt = loadmat(PATH_GT)['paviaU_gt']
# Window size used to compute features
WINDOWS_SHAPE = (5,5)
h = WINDOWS_SHAPE[0]//2
w = WINDOWS_SHAPE[1]//2
gt = gt[h:-h, w:-w]

for path in segmentations_paths:
    segmentation = np.load(path)
    assert segmentation.shape == gt.shape

#######################################################
#######################################################
# END OF HYPERPARAMETERS
#######################################################
#######################################################

# create folder to save analyses
folder = os.path.join(folder_result, 'analyses')
if not os.path.isdir(folder):
    os.mkdir(folder)

for path in segmentations_paths:
    segmentation = np.load(path)
    name = path.split('/')[-1].split('.')[0]
    folder_analyses = os.path.join(folder, name)
    if not os.path.isdir(folder_analyses):
        os.mkdir(folder_analyses)

    print('################################################')
    print('################################################')
    print()
    print('Results from', name)
    
    print('################################################')
    print('Supervised metric')
    print('################################################')
    old_segmentation = copy.deepcopy(segmentation)
    IoU, mIoU = compute_mIoU(old_segmentation, gt, np.unique(segmentation))
    print('mIoU before Hungarian algo=', mIoU)
    segmentation = assign_classes_segmentation_to_gt(segmentation, gt, normalize=False)
    IoU, mIoU = compute_mIoU(segmentation, gt, np.unique(segmentation))
    print('mIoU=', mIoU)
    classes = np.unique(segmentation).astype(np.int)
    temp = ''
    for i, class_ in enumerate(classes):
        temp += 'class '+  str(class_) + ': IoU=' + str(round(IoU[i], 2)) + ' '
    print(temp)

    print('################################################')
    print('Unsupervised metric')
    print('################################################')
    true = gt[gt!=0]
    pred = segmentation[gt!=0]
    AMI = adjusted_mutual_info_score(true, pred)
    ARI = adjusted_rand_score(true, pred)
    print('AMI=', AMI)
    print('ARI=', ARI)
    print('################################################')
    print('Plots saved in', folder_analyses)

    title = 'mIoU='+str(round(mIoU, 2))+' ARI='+str(round(ARI, 2))
    plot_segmentation(segmentation, classes=np.unique(gt).astype(np.int), title=title)
    plt.savefig(os.path.join(folder_analyses, 'segmentation.png'))
    plot_segmentation(gt, title='Ground truth')
    plt.savefig(os.path.join(folder_analyses, 'gt.png'))
    plot_TP_FP_FN_segmentation(segmentation, gt, folder_save=folder_analyses)
    
    plt.close('all')
