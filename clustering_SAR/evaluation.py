import matplotlib.pyplot as plt
import numpy as np
import os

def assign_classes_segmentation_to_gt(C, gt, normalize=False):
    """ A function that assigns the classes of the segmentation to the ground truth.
        Inputs:
            * C: segmented image.
            * gt: ground truth.
            * normalize: normalize each row of the cost matrix.
        Ouput:
            * segmented image with the right classes.
    """
    # import Hungarian algorithm
    from scipy.optimize import linear_sum_assignment
   
    classes = np.unique(gt)
    
    # if class 0 of gt is used for unnotated pixels then we make the classes of C start from 1
    if len(np.unique(gt)) == (len(np.unique(C))+1):
        classes = classes[1:]
    elif len(np.unique(gt)) != len(np.unique(C)):
        print('Error: wrong number of classes...')
        sys.exit(1)
    assert (classes == np.unique(C)).all()

    nb_classes = len(classes)

    cost_matrix = np.zeros((nb_classes, nb_classes))

    for i, class_gt in enumerate(classes):
        mask = (gt == class_gt)
        if normalize:
            nb_pixels = np.sum(mask)
        for j, class_C in enumerate(classes):
            cost = -np.sum(C[mask] == class_C)
            if normalize:
                cost /= nb_pixels
            cost_matrix[i, j] = cost
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    if len(np.unique(gt)) == (len(np.unique(C))+1):
        row_ind += 1
        col_ind += 1
    
    new_C = np.zeros(C.shape)
    for i, j in zip(col_ind, row_ind):
        new_C[C==i] = j

    return new_C


def plot_segmentation(C, aspect=1, classes=None, title=None):
    """ Plot a segmentation map.
        Inputs:
            * C: a (height, width) numpy array of integers (classes.
            * aspect: aspect ratio of the image.
            * classes: list of numbers of classes
            * title: string used for the title of the figure
    """
    if classes is not None:
        max_C, min_C = np.max(classes), np.min(classes)
    else:
        max_C, min_C = np.max(C), np.min(C) 

    #get discrete colormap
    cmap = plt.get_cmap('RdBu', max_C-min_C+1)
 
    # set limits .5 outside true range
    mat = plt.matshow(C, aspect=aspect, cmap=cmap, vmin=min_C-.5, vmax=max_C+.5)

    #tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(min_C,max_C+1))

    #title
    if title is not None:
        plt.title(title)


def save_segmentation(folder, filename, np_array):
    """ A function that saves a numpy array in a folder. The array and the folder are passed as arguments.
        Inputs:
            * folder: string.
            * filename: string.
            * np_array: numpy array to save.
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    np.save(path, np_array)


def plot_TP_FP_FN_segmentation(C, gt, aspect=1, folder_save=None):
    """ Plot True Positive, False Positive, False Negative for a segmetnation given a ground truth.
        Inputs:
            * C: a (height, width) numpy array of integers (classes).
            * gt: a (height, width) numpy array of integers (classes).
            * aspect: aspect ratio of the image.
            * folder_save: string representing the path of the folder where to save the plots. If not, plots are not saved.
    """
    # get classes
    classes = np.unique(C).astype(np.int)
    
    # get discrete colormap
    cmap = plt.get_cmap('RdBu', 4)
    
    for i in classes:
        to_plot = np.zeros(C.shape)
        # true positive
        mask = np.logical_and((C == i), (gt == i))
        to_plot[mask] = 3
 
        # false positive
        mask = np.logical_and(np.logical_and((C == i), (gt != i)), (gt != 0))
        to_plot[mask] = 2

        # false negative
        mask = np.logical_and((C != i), (gt == i))
        to_plot[mask] = 1
 
        # set limits .5 outside true range
        mat = plt.matshow(to_plot, aspect=aspect, cmap=cmap, vmin=-0.5, vmax=3.5)

        #tell the colorbar to tick at integers
        cax = plt.colorbar(mat)
        cax.set_ticks(np.arange(0, 4))
        cax.set_ticklabels(['Other', 'FN', 'FP', 'TP'])
 
        #title
        plt.title('Class '+str(i))

        if folder_save is not None:
            plt.savefig(os.path.join(folder_save, 'Class '+str(i)))


def compute_mIoU(C, gt, classes):
    """ A function that computes the mean of Intersection over Union between a segmented image (c) and a ground truth (gt). BE CAREFUL, 0 is considered as no annotation available.
        Inputs:
            * C: segmented image.
            * gt: ground truth.
            * classes: list of classes used to compute the mIOU
        Ouputs:
            * IoU, mIOU
    """
    IoU = list()
    for i in classes:
        inter = np.sum(np.logical_and((C == i), (gt == i)))
        union = np.sum(np.logical_and(np.logical_or((C == i), (gt == i)), (gt != 0)))
        IoU.append(inter/union)
    mIoU = np.mean(IoU)
    return IoU, mIoU
