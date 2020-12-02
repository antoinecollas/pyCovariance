import autograd.numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat
from sklearn.cluster import KMeans
import time

from pyCovariance import\
        K_means_datacube,\
        pca_image
from pyCovariance.evaluation import\
        assign_segmentation_classes_to_gt_classes,\
        compute_AMI,\
        compute_ARI,\
        compute_mIoU,\
        compute_OA,\
        plot_segmentation,\
        plot_TP_FP_FN_segmentation,\
        save_segmentation


class Dataset():
    def __init__(self, name):
        self.name = name
        if name == 'Pavia':
            self.path = 'data/Pavia/PaviaU.mat'
            self.key_dict = 'paviaU'
            self.path_gt = 'data/Pavia/PaviaU_gt.mat'
            self.key_dict_gt = 'paviaU_gt'
            self.resolution = [1.3, 1.3]  # resolution in meters
            self.dimension = 103
        elif name == 'Indian_Pines':
            self.path = 'data/Indian_Pines/Indian_pines_corrected.mat'
            self.key_dict = 'indian_pines_corrected'
            self.path_gt = 'data/Indian_Pines/Indian_pines_gt.mat'
            self.key_dict_gt = 'indian_pines_gt'
            self.resolution = [1.3, 1.3]  # resolution in meters
            self.dimension = 200
        else:
            print(name)
            raise NotImplementedError


class HyperparametersKMeans():
    def __init__(
        self,
        crop_image,
        enable_multi,
        pca,
        nb_bands_to_select,
        mask,
        window_size,
        features,
        nb_init,
        nb_iter_max,
        eps
    ):
        '''
            Hyperparameters used by function "K_means_hyperspectral_image".
            Inputs:
                crop_image: bool. If true, it crops image before
                    applying K means.
                enable_multi: bool. If true, K means uses multi processing.
                pca: bool. If true, pca is applied to reduce dimention
                    of pixels before applying K means.
                nb_bands_to_select: int. Number of dimentions to keep
                    (after a pca or dimentions are randomly chosen).
                mask: bool. If true it clusters only
                    where there is a ground truth.
                window_size: int. Number of pixels of height and
                    width in the window.
                features: Feature to use to cluster.
                nb_init: int. Number of initialisations.
                    Best clustering is kept.
                nb_iter_max: int. Maximum number of iterations
                    done by K means.
                eps: float. Stopping criteria.
        '''
        # image
        self.crop_image = crop_image
        if crop_image:
            self.size_crop = 50

        # multi processing
        self.enable_multi = enable_multi
        self.nb_threads_rows = os.cpu_count()//2
        self.nb_threads_columns = 2
        self.nb_threads = os.cpu_count()

        # preprocessing
        self.pca = pca
        self.nb_bands_to_select = nb_bands_to_select
        self.mask = mask

        # features
        self.window_size = window_size
        self.features = features

        # K-means
        self.nb_init = nb_init
        self.nb_iter_max = nb_iter_max
        self.eps = eps


def K_means_hyperspectral_image(dataset, hyperparams):
    t_beginning = time.time()

    print("###################### PREPROCESSING ######################")
    # load image and gt
    image = loadmat(dataset.path)[dataset.key_dict]
    gt = loadmat(dataset.path_gt)[dataset.key_dict_gt]
    gt = gt.astype(np.int64)
    gt -= 1
    print('Crop image:', hyperparams.crop_image)
    if hyperparams.crop_image:
        center = np.array(image.shape[0:2])//2
        half_height = hyperparams.size_crop//2
        half_width = hyperparams.size_crop//2
        image = image[center[0]-half_height:center[0]+half_height,
                      center[1]-half_width:center[1]+half_width]
        gt = gt[center[0]-half_height:center[0]+half_height,
                center[1]-half_width:center[1]+half_width]
    nb_classes = len(np.unique(gt)) - 1

    # center image globally
    mean = np.mean(image, axis=0)
    image = image - mean
    # check pixels are centered
    assert (np.abs(np.mean(image, axis=0)) < 1e-9).all()

    # pca
    if hyperparams.pca:
        image = pca_image(image, hyperparams.nb_bands_to_select)
    print('PCA:', hyperparams.pca)

    n_r, n_c, p = image.shape
    print('image.shape:', image.shape)

    # mask
    if hyperparams.mask:
        mask = (gt >= 0)
    else:
        mask = None

    if hyperparams.features == 'sklearn':
        h = w = hyperparams.window_size//2
        X = image[h:-h, w:-w, :]
        mask = mask[h:-h, w:-w]
        if mask is not None:
            X = X[mask]
        else:
            X = X.reshape((-1, hyperparams.nb_bands_to_select))
        temp = KMeans(
            n_clusters=nb_classes,
            n_init=hyperparams.nb_init,
            max_iter=hyperparams.nb_iter_max,
            tol=hyperparams.eps
        ).fit_predict(X)
        temp = temp+1
        if mask is not None:
            C = np.zeros((n_r-2*w, n_c-2*w))
            C[mask] = temp
        else:
            C = temp
    else:
        C = K_means_datacube(
            image,
            mask,
            hyperparams.features,
            hyperparams.window_size,
            nb_classes,
            hyperparams.nb_init,
            hyperparams.nb_iter_max,
            hyperparams.eps,
            hyperparams.enable_multi,
            hyperparams.nb_threads_rows,
            hyperparams.nb_threads_columns
        )
        C = C.squeeze()
    C = C.astype(np.int)

    t_end = time.time()
    print('TOTAL TIME ELAPSED:', round(t_end-t_beginning, 1), 's')

    return C


def evaluate_and_save_clustering(
    segmentation,
    dataset,
    hyperparams,
    folder,
    prefix_filename
):
    print('###################### EVALUATION ######################')

    # ground truth path
    gt = loadmat(dataset.path_gt)[dataset.key_dict_gt]
    if hyperparams.crop_image:
        center = np.array(gt.shape[0:2])//2
        half_height = hyperparams.size_crop//2
        half_width = hyperparams.size_crop//2
        gt = gt[center[0]-half_height:center[0]+half_height,
                center[1]-half_width:center[1]+half_width]
    h = (gt.shape[0]-segmentation.shape[0])//2
    w = (gt.shape[1]-segmentation.shape[1])//2
    if (h > 0) and (w > 0):
        gt = gt[h:-h, w:-w]

    assert segmentation.shape == gt.shape,\
           'segmentation.shape:' + str(segmentation.shape) +\
           ', gt.shape:'+str(gt.shape)

    # create folders
    folder_npy = os.path.join(folder, 'npy')
    if not os.path.isdir(folder_npy):
        os.makedirs(folder_npy, exist_ok=True)
    folder_segmentation = os.path.join(folder, 'segmentations')
    if not os.path.isdir(folder_segmentation):
        os.makedirs(folder_segmentation, exist_ok=True)
    f_name = prefix_filename+'_'+str(hyperparams.features)
    folder_detailed_analyses = os.path.join(folder,
                                            'detailed_analyses', f_name)
    if not os.path.isdir(folder_detailed_analyses):
        os.makedirs(folder_detailed_analyses, exist_ok=True)

    segmentation = assign_segmentation_classes_to_gt_classes(
        segmentation, gt, normalize=False)
    f_name = prefix_filename + '_K_means_' + str(hyperparams.features)
    save_segmentation(folder_npy, f_name, segmentation)

    # mIoU
    IoU, mIoU = compute_mIoU(segmentation, gt)
    mIoU = round(mIoU, 2)
    temp = 'IoU:'
    for i in range(len(IoU)):
        temp += ' class ' + str(i + 1) + ': ' + str(round(IoU[i], 2))
    print(temp)
    print('mIoU=', mIoU)

    # OA
    OA = compute_OA(segmentation, gt)
    OA = round(OA, 2)
    print('OA=', OA)

    # AMI
    AMI = compute_AMI(segmentation, gt)
    AMI = round(AMI, 2)
    print('AMI=', AMI)

    # ARI
    ARI = compute_ARI(segmentation, gt)
    ARI = round(ARI, 2)
    print('ARI=', ARI)

    plot_segmentation(gt, title='Ground truth')
    plt.savefig(os.path.join(folder_segmentation, 'gt'))

    title = 'mIoU='+str(round(mIoU, 2))+' OA='+str(round(OA, 2))
    classes = np.unique(gt).astype(np.int)
    plot_segmentation(segmentation, classes=classes, title=title)
    f_name = prefix_filename + '_K_means_' + str(hyperparams.features)
    plt.savefig(os.path.join(folder_segmentation, f_name))

    plot_TP_FP_FN_segmentation(segmentation, gt,
                               folder_save=folder_detailed_analyses)

    plt.close('all')

    return mIoU, OA
