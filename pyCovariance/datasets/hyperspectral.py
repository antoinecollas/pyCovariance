import autograd.numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans
import time
import wget

from ..cluster_datacube import K_means_datacube
from ..pca import pca_image
from ..evaluation import\
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
        self.size_crop = 30
        if name == 'Pavia':
            self.url = 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat'
            self.url_gt =\
                'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'
            self.path = 'data/Pavia/PaviaU.mat'
            self.key_dict = 'paviaU'
            self.path_gt = 'data/Pavia/PaviaU_gt.mat'
            self.key_dict_gt = 'paviaU_gt'
            self.resolution = [1.3, 1.3]  # resolution in meters
            self.dimension = 103
        elif name == 'Indian_Pines':
            self.url =\
                ('http://www.ehu.eus/ccwintco/uploads/6/67/'
                 'Indian_pines_corrected.mat')
            self.url_gt =\
                'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
            self.path = 'data/Indian_Pines/Indian_pines_corrected.mat'
            self.key_dict = 'indian_pines_corrected'
            self.path_gt = 'data/Indian_Pines/Indian_pines_gt.mat'
            self.key_dict_gt = 'indian_pines_gt'
            self.resolution = [1.3, 1.3]  # resolution in meters
            self.dimension = 200
        else:
            print(name)
            raise NotImplementedError
        self.download()

    def download(self):
        if not os.path.exists(self.path):
            _dir = os.path.dirname(self.path)
            os.makedirs(_dir, exist_ok=True)
            print('\n Download', self.name, 'dataset.')
            wget.download(self.url, _dir)
        if not os.path.exists(self.path_gt):
            _dir = os.path.dirname(self.path_gt)
            os.makedirs(_dir, exist_ok=True)
            print('\n Download', self.name, 'ground truth.')
            wget.download(self.url_gt, _dir)

    def load(self, crop_image):
        image = loadmat(self.path)[self.key_dict]
        gt = loadmat(self.path_gt)[self.key_dict_gt]
        gt = gt.astype(np.int64)

        # 'no class' has label -1
        gt -= 1

        if crop_image:
            center = np.array(image.shape[0:2])//2
            half_height = self.size_crop//2
            half_width = self.size_crop//2
            image = image[center[0]-half_height:center[0]+half_height,
                          center[1]-half_width:center[1]+half_width]
            gt = gt[center[0]-half_height:center[0]+half_height,
                    center[1]-half_width:center[1]+half_width]

        return image, gt


class HyperparametersKMeans():
    def __init__(
        self,
        crop_image,
        nb_threads,
        pca,
        nb_bands_to_select,
        mask,
        window_size,
        feature,
        nb_init,
        nb_iter_max,
        eps
    ):
        '''
            Hyperparameters used by function "K_means_hyperspectral_image".
            Inputs:
                crop_image: bool. If true, it crops image before
                    applying K means.
                    nb_threads: int. Number of threads for parallelisation.
                pca: bool. If true, pca is applied to reduce dimention
                    of pixels before applying K means.
                nb_bands_to_select: int. Number of dimentions to keep
                    (after a pca or dimentions are randomly chosen).
                mask: bool. If true it clusters only
                    where there is a ground truth.
                window_size: int. Number of pixels of height and
                    width in the window.
                feature: Feature to use to cluster.
                nb_init: int. Number of initialisations.
                    Best clustering is kept.
                nb_iter_max: int. Maximum number of iterations
                    done by K means.
                eps: float. Stopping criteria.
        '''
        # image
        self.crop_image = crop_image

        # multi processing
        if nb_threads > 1:
            self.nb_threads_rows = nb_threads//2
            self.nb_threads_columns = 2
        else:
            self.nb_threads_rows = 1
            self.nb_threads_columns = 1

        # preprocessing
        self.pca = pca
        self.nb_bands_to_select = nb_bands_to_select
        self.mask = mask

        # feature
        self.window_size = window_size
        self.feature = feature

        # K-means
        self.nb_init = nb_init
        self.nb_iter_max = nb_iter_max
        self.eps = eps


def K_means_hyperspectral_image(dataset, hyperparams):
    t_beginning = time.time()

    print("###################### PREPROCESSING ######################")
    # load image and gt
    image, gt = dataset.load(hyperparams.crop_image)
    print('Crop image:', hyperparams.crop_image)

    nb_classes = np.sum(np.unique(gt) >= 0)

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

    if hyperparams.feature == 'sklearn':
        h = w = hyperparams.window_size//2
        X = image[h:-h, w:-w, :]
        mask = mask[h:-h, w:-w]
        if mask is not None:
            X = X[mask]
        else:
            X = X.reshape((-1, hyperparams.nb_bands_to_select))
        sklearn_K_means = KMeans(
            n_clusters=nb_classes,
            n_init=hyperparams.nb_init,
            max_iter=hyperparams.nb_iter_max,
            tol=hyperparams.eps
        )
        temp = sklearn_K_means.fit_predict(X)
        if mask is not None:
            C = np.zeros((n_r-2*w, n_c-2*w)) - 1
            C[mask] = temp
        else:
            C = temp
        criterion_values = [sklearn_K_means.inertia_]
    else:
        C, criterion_values = K_means_datacube(
            image,
            mask,
            hyperparams.feature,
            hyperparams.window_size,
            nb_classes,
            hyperparams.nb_init,
            hyperparams.nb_iter_max,
            hyperparams.eps,
            hyperparams.nb_threads_rows,
            hyperparams.nb_threads_columns
        )

    t_end = time.time()
    print('TOTAL TIME ELAPSED:', round(t_end-t_beginning, 1), 's')

    return C, criterion_values


def evaluate_and_save_clustering(
    segmentation,
    dataset,
    hyperparams,
    folder,
    prefix_filename
):
    print('###################### EVALUATION ######################')

    _, gt = dataset.load(hyperparams.crop_image)

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
    f_name = prefix_filename+'_'+str(hyperparams.feature)
    folder_detailed_analyses = os.path.join(folder,
                                            'detailed_analyses', f_name)
    if not os.path.isdir(folder_detailed_analyses):
        os.makedirs(folder_detailed_analyses, exist_ok=True)

    segmentation = assign_segmentation_classes_to_gt_classes(
        segmentation, gt, normalize=False)
    f_name = prefix_filename + '_K_means_' + str(hyperparams.feature)
    save_segmentation(folder_npy, f_name, segmentation)

    # mIoU
    IoU, mIoU = compute_mIoU(segmentation, gt)
    mIoU = round(mIoU, 2)
    temp = 'IoU:'
    for i in range(len(IoU)):
        temp += ' class ' + str(i + 1) +\
                ': ' + str(round(IoU[i], 2))
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

    plot_segmentation(gt + 1, title='Ground truth')
    plt.savefig(os.path.join(folder_segmentation, 'gt'))

    title = 'mIoU='+str(round(mIoU, 2))+' OA='+str(round(OA, 2))
    plot_segmentation(segmentation + 1, title=title)
    f_name = prefix_filename + '_K_means_' + str(hyperparams.feature)
    plt.savefig(os.path.join(folder_segmentation, f_name))

    classes_labels = np.unique(gt[gt >= 0]) + 1
    plot_TP_FP_FN_segmentation(segmentation, gt, classes_labels=classes_labels,
                               folder_save=folder_detailed_analyses)

    plt.close('all')

    return mIoU, OA
