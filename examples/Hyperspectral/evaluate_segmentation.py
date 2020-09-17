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

from clustering_SAR.evaluation import assign_classes_segmentation_to_gt, compute_mIoU, compute_OA, plot_segmentation, plot_TP_FP_FN_segmentation

from examples.Hyperspectral.k_means import Dataset

#######################################################
#######################################################
# BEGINNING OF HYPERPARAMETERS
#######################################################
#######################################################

DATASET_LIST = ['Indian_Pines', 'Pavia']

# Window size used to compute features
WINDOWS_SHAPE = (7, 7)

#######################################################
#######################################################
# END OF HYPERPARAMETERS
#######################################################
#######################################################

for dataset_name in DATASET_LIST:
    dataset = Dataset(dataset_name)
    print()
    print()
    print()
    print('################################################')
    print('Dataset', dataset.name) 
    print('################################################')
    
    # segmentation path (by default: get last folder)
    folder_regex = os.path.join('results', dataset.name, '*')
    folders_result = glob.glob(folder_regex)
    if len(folders_result)==0:
        print('Warning: no results for dataset', dataset.name)
        continue
    folders_result.sort()
    folder_result = folders_result[-1]
    print('Folder used to get segmentations:', folder_result)
    segmentations_paths = glob.glob(folder_result+'/*.npy')
    segmentations_paths.sort()

    if len(segmentations_paths) == 0:
        print('No npy files found ...')
        sys.exit(1)

    # ground truth path
    gt = loadmat(dataset.path_gt)[dataset.key_dict_gt]

    h = WINDOWS_SHAPE[0]//2
    w = WINDOWS_SHAPE[1]//2
    gt = gt[h:-h, w:-w]

    for path in segmentations_paths:
        segmentation = np.load(path)
        assert segmentation.shape == gt.shape, 'segmentation.shape:'+str(segmentation.shape)+', gt.shape:'+str(gt.shape)

    # create folder to save analyses
    folder = os.path.join(folder_result, 'analyses')
    if not os.path.isdir(folder):
        os.mkdir(folder)

    for path in segmentations_paths:
        segmentation = np.load(path)
        name = path.split('/')[-1].split('.')[0]
        folder_detailed_analyses = os.path.join(folder, name)
        if not os.path.isdir(folder_detailed_analyses):
            os.mkdir(folder_detailed_analyses)

        print('################################################')
        print()
        print('Results from', name)
        
        print('################################################')
        print('Metrics:')
        print('################################################')
        old_segmentation = copy.deepcopy(segmentation)
        segmentation = assign_classes_segmentation_to_gt(segmentation, gt, normalize=False)
        
        IoU, mIoU = compute_mIoU(segmentation, gt)
        mIoU = round(mIoU, 2)
        temp = 'IoU:'
        for i in range(len(IoU)):
            temp += ' class '+  str(i+1) + ': ' + str(round(IoU[i], 2))
        print(temp)
        print('mIoU=', mIoU)
        
        OA = compute_OA(segmentation, gt)
        OA = round(OA, 2)
        print('OA=', OA)

        true = gt[gt!=0]
        pred = segmentation[gt!=0]
        AMI = adjusted_mutual_info_score(true, pred)
        AMI = round(AMI, 2)
        ARI = adjusted_rand_score(true, pred)
        ARI = round(ARI, 2)
        print('AMI=', AMI)
        print('ARI=', ARI)
        print('Plots saved in', folder_detailed_analyses)

        title = 'mIoU='+str(round(mIoU, 2))+' OA='+str(round(OA, 2))
        plot_segmentation(segmentation, classes=np.unique(gt).astype(np.int), title=title)
        plt.savefig(os.path.join(folder, 'segmentation_'+name+'.png'))
        plot_segmentation(gt, title='Ground truth')
        plt.savefig(os.path.join(folder_detailed_analyses, 'gt.png'))
        plot_TP_FP_FN_segmentation(segmentation, gt, folder_save=folder_detailed_analyses)
        
        plt.close('all')
