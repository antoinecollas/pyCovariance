import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import tikzplotlib

from pyCovariance.evaluation import create_directory

from pyCovariance.features import\
        covariance,\
        tau_UUH
# center_euclidean,\
# covariance_texture,\
# mean_vector_euclidean,\
# subspace_SCM,\

from pyCovariance.datasets.hyperspectral import\
        Dataset,\
        evaluate_and_save_clustering,\
        HyperparametersKMeans,\
        K_means_hyperspectral_image


def main(
    dataset,
    crop_image,
    border_size,
    pairs_window_size_n_bands,
    mask,
    features_list,
    n_experiments,
    n_init,
    max_iter,
    n_jobs,
    export_tex=False,
    verbose=True
):

    # folder path to save files
    folder_main = create_directory(dataset.name)

    # get biggest window size to eliminate borders of the image in the mask
    ws = list()
    for w, _ in pairs_window_size_n_bands:
        ws.append(w)
    ws.sort()
    max_w = ws[-1]//2
    border_size = max(border_size, max_w)

    hp = HyperparametersKMeans(
        crop_image=crop_image,
        n_jobs=n_jobs,
        pca=None,
        nb_bands_to_select=None,
        mask=mask,
        border_size=border_size,
        window_size=None,
        feature=None,
        init='k-means++',
        n_init=n_init,
        max_iter=max_iter,
        eps=1e-3
    )

    pairs_w_p = pairs_window_size_n_bands

    # check that there is same number of features for all (w, p) pairs
    n_features = len(features_list[0])
    for i in range(len(pairs_w_p)):
        assert n_features == len(features_list[i])

    matrix_mIoUs = np.zeros((len(pairs_w_p), n_features))
    matrix_OAs = np.zeros((len(pairs_w_p), n_features))
    matrix_mIoUs_std = np.zeros((len(pairs_w_p), n_features))
    matrix_OAs_std = np.zeros((len(pairs_w_p), n_features))

    for i, (w_size, p) in enumerate(pairs_w_p):
        if verbose:
            print()
            print('########################################################')
            print('w_size =', w_size)
            print('p =', p)
            print('########################################################')

        hp.window_size = w_size
        hp.nb_bands_to_select = p

        features = features_list[i]

        prefix = 'w' + str(w_size) + '_p' + str(p)
        folder = os.path.join(folder_main, prefix)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        folder_criteria = os.path.join(folder, 'K-means_criteria')
        if not os.path.exists(folder_criteria):
            os.makedirs(folder_criteria, exist_ok=True)

        # K means and evaluations
        mIoUs, OAs = list(), list()
        mIoUs_std, OAs_std = list(), list()

        for j, feature in enumerate(features):
            hp.feature = feature

            if verbose:
                print()

            pattern = re.compile(r'tau_?\w*_UUH_?\w*|subspace_SCM')
            if type(hp.feature) is str:
                condition = pattern.match(hp.feature)
            else:
                N = w_size**2
                condition = pattern.match(str(hp.feature(p, N)))
            if condition:
                hp.pca = False
            else:
                hp.pca = True

            if verbose:
                if type(hp.feature) is str:
                    _str = hp.feature
                else:
                    _str = str(hp.feature(p, w_size**2))
                print('Feature:', _str)

            mIoU, OA = list(), list()

            if verbose:
                print('Crop image:', hp.crop_image)
                print('PCA:', hp.pca)

            # load image and gt
            image, gt = dataset.load(
                crop_image=hp.crop_image,
                pca=hp.pca,
                nb_bands_to_select=hp.nb_bands_to_select
            )

            if verbose:
                print('image.shape:', image.shape)
                print()

            for k in range(n_experiments):
                if verbose:
                    print('Experiment:', str(k+1) + '/' + str(n_experiments))

                C, criterion_values = K_means_hyperspectral_image(
                    image=image,
                    gt=gt,
                    hyperparams=hp,
                    verbose=verbose
                )

                prefix_f_name = str(j) + '_exp_' + str(k)
                tmp_mIoU, tmp_OA = evaluate_and_save_clustering(
                    segmentation=C,
                    dataset=dataset,
                    hyperparams=hp,
                    folder=folder,
                    prefix_filename=prefix_f_name,
                    export_pgf=export_tex,
                    verbose=False
                )
                if verbose:
                    print('mIoU=', round(tmp_mIoU, 3))
                    print('OA=', round(tmp_OA, 3))
                mIoU.append(tmp_mIoU)
                OA.append(tmp_OA)

                matplotlib.use('Agg')

                # save plot of within classes variances
                fig, ax = plt.subplots(1)
                for c_value in criterion_values:
                    x = list(range(len(c_value)))
                    plt.plot(x, c_value, '+--')
                plt.ylabel('sum of within-classes variances')
                if callable(feature):
                    N = w_size ** 2
                    feature = feature(p, N)
                plt.title('Criterion values of ' + str(feature) + ' feature.')
                temp = str(j) + '_criterion_' + str(feature) + '_exp_' + str(k)
                path = os.path.join(folder_criteria, temp)
                if export_tex:
                    tikzplotlib.save(path)
                plt.savefig(path)

                plt.close('all')

                if verbose:
                    print()

            mIoU_mean = round(np.mean(mIoU), 4)
            mIoU_std = round(np.std(mIoU), 4)
            mIoUs.append(mIoU_mean)
            mIoUs_std.append(mIoU_std)
            OA_mean = round(np.mean(OA), 4)
            OA_std = round(np.std(OA), 4)
            OAs.append(OA_mean)
            OAs_std.append(OA_std)
            if verbose:
                print('Final mIoU=', mIoU_mean, '+-', mIoU_std)
                print('Final OA=', OA_mean, '+-', OA_std)

        matrix_mIoUs[i, :] = mIoUs
        matrix_OAs[i, :] = OAs
        matrix_mIoUs_std[i, :] = mIoUs_std
        matrix_OAs_std[i, :] = OAs_std

    # comparison between models for a (w, p) fixed
    for i, (w_size, p) in enumerate(pairs_w_p):
        features_str = list()
        for j, feature in enumerate(features_list[i]):
            if type(feature) is not str:
                feature = feature(p, w_size**2)
            features_str.append(str(j) + '_' + str(feature))

        prefix = 'w' + str(w_size) + '_p' + str(p)
        folder = os.path.join(folder_main, prefix)

        # Bar plot of mIoUs
        fig, ax = plt.subplots(1)
        ax.bar(features_str, matrix_mIoUs[i, :],
               yerr=matrix_mIoUs_std[i, :], align='center')
        ax.set_ylim(0, 1)
        plt.ylabel('mIoU')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        path = os.path.join(folder, 'mIoU_' + prefix)
        if export_tex:
            tikzplotlib.save(path)
        plt.savefig(path)

        # Bar plot of OAs
        fig, ax = plt.subplots(1)
        ax.bar(features_str, matrix_OAs[i, :],
               yerr=matrix_OAs_std[i, :], align='center')
        ax.set_ylim(0, 1)
        plt.ylabel('OA')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        path = os.path.join(folder, 'OA_' + prefix)
        if export_tex:
            tikzplotlib.save(path)
        plt.savefig(path)

        plt.close('all')

    # comparison between pairs of (w, p) for a fixed model
    for i, feature in enumerate(features_list[0]):
        pairs_str = list()
        for j, (w_size, p) in enumerate(pairs_w_p):
            pairs_str.append('w' + str(w_size) + '_p' + str(p))

        folder = os.path.join(folder_main)

        # Bar plot of mIoUs
        fig, ax = plt.subplots(1)
        ax.bar(pairs_str, matrix_mIoUs[:, i].reshape(-1),
               yerr=matrix_mIoUs_std[:, i].reshape(-1), align='center')
        ax.set_ylim(0, 1)
        plt.ylabel('mIoU')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        if type(feature) is not str:
            feature = feature(p, w_size**2)
        path = os.path.join(folder, str(i) + '_mIoU_' + str(feature))
        if export_tex:
            tikzplotlib.save(path)
        plt.savefig(path)

        # Bar plot of OAs
        fig, ax = plt.subplots(1)
        ax.bar(pairs_str, matrix_OAs[:, i].reshape(-1),
               yerr=matrix_mIoUs_std[:, i].reshape(-1), align='center')
        ax.set_ylim(0, 1)
        plt.ylabel('OA')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        path = os.path.join(folder, str(i) + '_OA_' + str(feature))
        if export_tex:
            tikzplotlib.save(path)
        plt.savefig(path)

        plt.close('all')


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    print('seed:', seed)

    pairs_w_k, features_list = list(), list()

    #####
    w, k = 5, 5
    pairs_w_k.append((w, k))

    # w=5, k=5, estimate_sigma=True
    C_tau = 47.27
    C_U = 5.14

    features_list.append([
        covariance(),
        tau_UUH(k, weights=(0.1/C_tau, 0.9/C_U), estimate_sigma=True),
    ])

    #####
    w, k = 7, 5
    pairs_w_k.append((w, k))

    # w=7, k=5, estimate_sigma=True
    C_tau = 92.10
    C_U = 4.80

    features_list.append([
        covariance(),
        tau_UUH(k, weights=(0.1/C_tau, 0.9/C_U), estimate_sigma=True),
    ])

    #####
    w, k = 9, 5
    pairs_w_k.append((w, k))

    # w=9, k=5, estimate_sigma=True
    C_tau = 157.90
    C_U = 4.58

    features_list.append([
        covariance(),
        tau_UUH(k, weights=(0.1/C_tau, 0.9/C_U), estimate_sigma=True),
    ])

    #####
    w, k = 11, 5
    pairs_w_k.append((w, k))

    # w=11, k=5, estimate_sigma=True
    C_tau = 239.33
    C_U = 4.37

    features_list.append([
        covariance(),
        tau_UUH(k, weights=(0.1/C_tau, 0.9/C_U), estimate_sigma=True),
    ])

    dataset_name = 'Indian_Pines'  # or 'Pavia' or 'Salinas'
    # border_size: discard 5 pixels around the image
    # used to compare with
    # different windows 5x5 vs 7x7 vs 9x9 vs 11x11
    # 11//2 == 5
    border_size = 5
    dataset = Dataset(dataset_name)
    main(
        dataset=dataset,
        crop_image=False,
        border_size=border_size,
        pairs_window_size_n_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        n_experiments=10,
        n_init=10,
        max_iter=100,
        n_jobs=os.cpu_count(),
        export_tex=True
    )
