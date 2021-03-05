import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import tikzplotlib

from pyCovariance.evaluation import create_directory

from pyCovariance.features import\
        center_euclidean,\
        covariance,\
        covariance_texture,\
        mean_vector_euclidean,\
        subspace_SCM,\
        tau_UUH

from pyCovariance.datasets.hyperspectral import\
        Dataset,\
        evaluate_and_save_clustering,\
        HyperparametersKMeans,\
        K_means_hyperspectral_image


def main(
    dataset,
    crop_image,
    border_size,
    pairs_window_size_nb_bands,
    mask,
    features_list,
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
    for w, _ in pairs_window_size_nb_bands:
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

    pairs_w_p = pairs_window_size_nb_bands

    # check that there is same number of features for all (w, p) pairs
    nb_features = len(features_list[0])
    for i in range(len(pairs_w_p)):
        assert nb_features == len(features_list[i])

    matrix_mIoUs = np.zeros((len(pairs_w_p), nb_features))
    matrix_OAs = np.zeros((len(pairs_w_p), nb_features))

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
        mIoUs = list()
        OAs = list()

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

            C, criterion_values = K_means_hyperspectral_image(
                dataset,
                hp,
                verbose
            )

            prefix_f_name = str(j)
            mIoU, OA = evaluate_and_save_clustering(
                segmentation=C,
                dataset=dataset,
                hyperparams=hp,
                folder=folder,
                prefix_filename=prefix_f_name,
                export_pgf=export_tex,
                verbose=verbose
            )
            mIoUs.append(mIoU)
            OAs.append(OA)

            matplotlib.use('Agg')

            # save plot of within classes variances
            fig, ax = plt.subplots(1)
            for c_value in criterion_values:
                x = list(range(len(c_value)))
                plt.plot(x, c_value, '+--')
            plt.ylabel('sum of within-classes variances')
            if type(feature) is not str:
                N = w_size ** 2
                feature = feature(p, N)
            plt.title('Criterion values of ' + str(feature) + ' feature.')
            temp = str(j) + '_criterion_' + str(feature)
            path = os.path.join(folder_criteria, temp)
            if export_tex:
                tikzplotlib.save(path)
            plt.savefig(path)

            plt.close('all')

        matrix_mIoUs[i, :] = mIoUs
        matrix_OAs[i, :] = OAs

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
        ax.bar(features_str, matrix_mIoUs[i, :], align='center')
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
        ax.bar(features_str, matrix_OAs[i, :], align='center')
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
        ax.bar(pairs_str, matrix_mIoUs[:, i].reshape(-1), align='center')
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
        ax.bar(pairs_str, matrix_OAs[:, i].reshape(-1), align='center')
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

    pairs_w_k = [(7, 5)]
    C_tau_1 = 92.77
    C_U_1 = 4.81
    C_tau_2 = 90.85
    C_U_2 = 4.80
    features_list = list()
    for _, k in pairs_w_k:
        features_list.append([
            'sklearn',
            center_euclidean(),
            mean_vector_euclidean(),
            covariance(),
            covariance_texture(),
            subspace_SCM(k),
            tau_UUH(k, weights=(0, 1), estimate_sigma=True),
            tau_UUH(k, weights=(0.1/C_tau_1, 0.9/C_U_1), estimate_sigma=True),
            tau_UUH(k, weights=(0.2/C_tau_1, 0.8/C_U_1), estimate_sigma=True),
            tau_UUH(k, weights=(0.3/C_tau_1, 0.7/C_U_1), estimate_sigma=True),
            tau_UUH(k, weights=(0.4/C_tau_1, 0.6/C_U_1), estimate_sigma=True),
            tau_UUH(k, weights=(0.5/C_tau_1, 0.5/C_U_1), estimate_sigma=True),
            tau_UUH(k, weights=(0, 1), estimate_sigma=False),
            tau_UUH(k, weights=(0.1/C_tau_2, 0.9/C_U_2), estimate_sigma=False),
            tau_UUH(k, weights=(0.2/C_tau_2, 0.8/C_U_2), estimate_sigma=False),
            tau_UUH(k, weights=(0.3/C_tau_2, 0.7/C_U_2), estimate_sigma=False),
            tau_UUH(k, weights=(0.4/C_tau_2, 0.6/C_U_2), estimate_sigma=False),
            tau_UUH(k, weights=(0.5/C_tau_2, 0.5/C_U_2), estimate_sigma=False)
        ])

    dataset_name = 'Indian_Pines'  # or 'Pavia' or 'Salinas'
    # border_size: discard 4 pixels around the image
    # used to compare with
    # different windows 5x5 vs 7x7 vs 9x9
    # 9//2 == 4
    border_size = 4
    dataset = Dataset(dataset_name)
    main(
        dataset=dataset,
        crop_image=False,
        border_size=border_size,
        pairs_window_size_nb_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        n_init=10,
        max_iter=100,
        n_jobs=os.cpu_count(),
        export_tex=True
    )
