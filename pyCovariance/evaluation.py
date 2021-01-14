import autograd.numpy as np
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import os
from sklearn.metrics import\
        adjusted_mutual_info_score,\
        adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import tikzplotlib


def _get_classes(C, gt):
    classes_C = np.unique(C).astype(np.int)
    classes_gt = np.unique(gt).astype(np.int)
    classes = np.array(list(set().union(classes_C, classes_gt)))
    classes = classes.astype(np.int64)
    classes = classes[classes >= 0]
    classes = np.sort(classes)
    return classes


def assign_segmentation_classes_to_gt_classes(C, gt, normalize=False):
    """ A function that assigns the classes of the segmentation to
        the classes of the ground truth.
        BE CAREFUL : negative classes are always ignored, both in C and gt.
        Inputs:
            * C: segmented image.
            * gt: ground truth.
            * normalize: normalize each row of the cost matrix.
        Ouput:
            * segmented image with the right classes.
    """
    # get classes
    classes_gt = _get_classes(gt, gt)
    classes_C = _get_classes(C, C)
    assert len(classes_gt) == len(classes_C)
    nb_classes = len(classes_gt)

    cost_matrix = np.zeros((nb_classes, nb_classes))

    for i, class_gt in enumerate(classes_gt):
        mask = (gt == class_gt)
        if normalize:
            nb_pixels = np.sum(mask)
        for j, class_C in enumerate(classes_C):
            cost = -np.sum(C[mask] == class_C)
            if normalize:
                cost /= nb_pixels
            cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    row_ind = classes_gt[row_ind]
    col_ind = classes_C[col_ind]
    new_C = deepcopy(C)
    for i, j in zip(col_ind, row_ind):
        new_C[C == i] = j

    return new_C


def compute_mIoU(C, gt):
    """ A function that computes the mean of Intersection over Union between
    a segmented image (c) and a ground truth (gt).
    BE CAREFUL, negative values in gt are considered
    as no annotation available.
        Inputs:
            * C: segmented image.
            * gt: ground truth.
            * classes: list of classes used to compute the mIOU
        Ouputs:
            * IoU, mIOU
    """
    # get classes
    classes = _get_classes(C, gt)

    IoU = list()
    for i in classes:
        inter = np.sum((C == i) & (gt == i))
        union = np.sum(((C == i) | (gt == i)) & np.isin(gt, classes))
        IoU.append(inter/union)
    mIoU = np.mean(IoU)
    return IoU, mIoU


def compute_OA(C, gt):
    """ A function that computes the Overall Accuracy (OA) between
    a segmented image (c) and a ground truth (gt).
    BE CAREFUL, negative values in gt are considered
    as no annotation available.
        Inputs:
            * C: segmented image.
            * gt: ground truth.
        Ouputs:
            * OA
    """
    # get classes
    classes = _get_classes(C, gt)

    mask = np.isin(gt, classes)
    OA = np.sum(C[mask] == gt[mask]) / np.sum(mask)

    return OA


def compute_AMI(C, gt):
    """ A function that computes the adjusted mutual info between
    a segmented image (c) and a ground truth (gt).
    BE CAREFUL, negative values in gt are considered
    as no annotation available.
        Inputs:
            * C: segmented image.
            * gt: ground truth.
        Ouputs:
            * adjusted mutual info
    """
    # get classes
    classes = _get_classes(C, gt)

    mask = np.isin(gt, classes)
    AMI = adjusted_mutual_info_score(gt[mask], C[mask])

    return AMI


def compute_ARI(C, gt):
    """ A function that computes the adjusted rand score between
    a segmented image (c) and a ground truth (gt).
    BE CAREFUL, negative values in gt are considered
    as no annotation available.
        Inputs:
            * C: segmented image.
            * gt: ground truth.
        Ouputs:
            * adjusted rand score
    """
    # get classes
    classes = _get_classes(C, gt)

    mask = np.isin(gt, classes)
    ARI = adjusted_rand_score(gt[mask], C[mask])

    return ARI


def plot_segmentation(C, aspect=1, title=None):
    """ Plot a segmentation map.
        Inputs:
            * C: a (height, width) numpy array of integers (classes.
            * aspect: aspect ratio of the image.
            * title: string used for the title of the figure
    """
    max_C, min_C = np.max(C), np.min(C)

    # get discrete colormap
    cmap = plt.get_cmap('RdBu', max_C-min_C+1)

    # set limits .5 outside true range
    mat = plt.matshow(C, aspect=aspect, cmap=cmap,
                      vmin=min_C-.5, vmax=max_C+.5)

    # tell the colorbar to tick at integers
    plt.colorbar(mat, ticks=np.arange(min_C, max_C+1))

    # title
    if title is not None:
        plt.title(title)

    plt.grid(b=False)


def create_directory(name):
    """ A function that creates a directory to save results.
    The path has the 'results/name/date'.
    """
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join('results', name, date_str)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def save_segmentation(folder, filename, np_array):
    """ A function that saves a numpy array in a folder.
    The array and the folder are passed as arguments.
        Inputs:
            * folder: string.
            * filename: string.
            * np_array: numpy array to save.
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    np.save(path, np_array)


def plot_TP_FP_FN_segmentation(C, gt, aspect=1,
                               classes_labels=None, folder_save=None):
    """ Plot True Positive, False Positive, False Negative
    for a segmetnation given a ground truth.
    BE CAREFUL: negative classes are ignored
        Inputs:
            * C: a (height, width) numpy array of integers (classes).
            * gt: a (height, width) numpy array of integers (classes).
            * aspect: aspect ratio of the image.
            * classes_labels: list of numbers of classes to put in labels
            * folder_save: string of the path of the folder to save the plots.
            If not, plots are not saved.
    """
    # get classes
    classes = _get_classes(C, gt)

    if classes_labels is not None:
        assert len(classes) == len(classes_labels)

    # get discrete colormap
    cmap = plt.get_cmap('RdBu', 4)

    for i, k in zip(classes_labels, classes):
        to_plot = np.zeros(C.shape)
        # true positive
        mask = np.logical_and((C == k), (gt == k))
        to_plot[mask] = 3

        # false positive
        mask = np.logical_and(np.logical_and((C == k),
                                             (gt != k)), np.isin(gt, classes))
        to_plot[mask] = 2

        # false negative
        mask = np.logical_and((C != k), (gt == k))
        to_plot[mask] = 1

        # set limits .5 outside true range
        mat = plt.matshow(to_plot, aspect=aspect,
                          cmap=cmap, vmin=-0.5, vmax=3.5)

        # tell the colorbar to tick at integers
        cax = plt.colorbar(mat)
        cax.set_ticks(np.arange(0, 4))
        cax.set_ticklabels(['Other', 'FN', 'FP', 'TP'])

        # title
        if classes_labels is None:
            plt.title('Class '+str(int(k)))
        else:
            plt.title('Class '+str(int(i)))

        # remove grid
        plt.grid(b=False)

        if folder_save is not None:
            path = os.path.join(folder_save, 'Class_'+str(int(i))+'.png')
            plt.savefig(path)


def plot_Pauli_SAR(image, aspect=1):
    """ 1st dimension =HH, 2nd dimnension = HV, 3rd dimension=VV"""
    R = np.abs(image[:, :, 0] - image[:, :, 2])
    G = np.abs(image[:, :, 1])
    B = np.abs(image[:, :, 0] + image[:, :, 2])
    fig = plt.figure()
    RGB_image = np.stack([R, G, B], axis=2)
    RGB_image = RGB_image - np.min(RGB_image)
    RGB_image[RGB_image > 1] = 1
    plt.imshow(RGB_image, aspect=aspect)
    plt.axis('off')
    return fig


def save_figure(folder, figname):
    """ A function that save the current figure in '.png' and in '.tex'.
        Inputs:
            * folder: string of the folder's name where to save actual figure.
            * figname: string of the name of the figure to save.
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, figname)

    path_png = path + '.png'
    plt.savefig(path_png)

    path_tex = path + '.tex'
    tikzplotlib.save(path_tex)
