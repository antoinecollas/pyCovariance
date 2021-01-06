import autograd.numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import os
import re

from pyCovariance.features import\
        mean_pixel_euclidean,\
        pixel_euclidean,\
        covariance,\
        covariance_euclidean,\
        covariance_texture,\
        tau_UUH

from pyCovariance.datasets.hyperspectral import\
        Dataset,\
        evaluate_and_save_clustering,\
        HyperparametersKMeans,\
        K_means_hyperspectral_image


def main(
    dataset,
    crop_image,
    nb_threads,
    pairs_window_size_nb_bands,
    mask,
    features_list,
    nb_init,
    nb_iter_max
):
    matplotlib.use('Agg')

    # folder path to save files
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    folder_main = os.path.join('results', dataset.name, date_str)

    # get biggest window size to eliminate borders of the image in the mask
    ws = list()
    for w, _ in pairs_window_size_nb_bands:
        ws.append(w)
    ws.sort()
    max_w = ws[-1]

    hp = HyperparametersKMeans(
        crop_image=crop_image,
        nb_threads=nb_threads,
        pca=None,
        nb_bands_to_select=None,
        mask=mask,
        border_size=max_w,
        window_size=None,
        feature=None,
        nb_init=nb_init,
        nb_iter_max=nb_iter_max,
        eps=1e-3
    )

    pairs_w_p = pairs_window_size_nb_bands

    # check that there is smae number of features for all (w, p) pairs
    nb_features = len(features_list[0])
    for i in range(len(pairs_w_p)):
        assert nb_features == len(features_list[i])

    matrix_mIoUs = np.zeros((len(pairs_w_p), nb_features))
    matrix_OAs = np.zeros((len(pairs_w_p), nb_features))

    for i, (w_size, p) in enumerate(pairs_w_p):
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

            print()

            pattern = re.compile(r'tau_?\w*_UUH_?\w*')
            if pattern.match(str(hp.feature)):
                hp.pca = False
            else:
                hp.pca = True

            print('Feature:', str(hp.feature))

            C, criterion_values = K_means_hyperspectral_image(dataset, hp)

            prefix_f_name = str(j)
            mIoU, OA = evaluate_and_save_clustering(C, dataset,
                                                    hp, folder, prefix_f_name)
            mIoUs.append(mIoU)
            OAs.append(OA)

            # save plot of within classes variances
            fig, ax = plt.subplots(1)
            for c_value in criterion_values:
                x = list(range(len(c_value)))
                plt.plot(x, c_value, '+--')
            plt.ylabel('sum of within-classes variances')
            plt.title('Criterion values of ' + str(hp.feature) + ' feature.')
            temp = str(j) + '_criterion_' + str(hp.feature)
            path = os.path.join(folder_criteria, temp)
            plt.savefig(path)

            plt.close('all')

        matrix_mIoUs[i, :] = mIoUs
        matrix_OAs[i, :] = OAs

    # comparison between models for a (w, p) fixed
    for i, (w_size, p) in enumerate(pairs_w_p):
        features_str = list()
        for feature in features:
            features_str.append(str(feature))

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
        plt.savefig(path)

        # Bar plot of OAs
        fig, ax = plt.subplots(1)
        ax.bar(features_str, matrix_OAs[i, :], align='center')
        ax.set_ylim(0, 1)
        plt.ylabel('OA')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        path = os.path.join(folder, 'OA_' + prefix)
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
        path = os.path.join(folder, str(i) + '_mIoU_' + str(feature))
        plt.savefig(path)

        # Bar plot of OAs
        fig, ax = plt.subplots(1)
        ax.bar(pairs_str, matrix_OAs[:, i].reshape(-1), align='center')
        ax.set_ylim(0, 1)
        plt.ylabel('OA')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        path = os.path.join(folder, str(i) + '_OA_' + str(feature))
        plt.savefig(path)

        plt.close('all')


if __name__ == '__main__':
    def get_features(pairs_w_k, p):
        features_list = list()
        for w, k in pairs_w_k:
            features_list.append([
                'sklearn',
                pixel_euclidean(k),
                mean_pixel_euclidean(k),
                covariance_euclidean(k),
                covariance(k),
                covariance_texture(k, w*w),
                tau_UUH(w*w, p, k),
                tau_UUH(w*w, p, k, weights=(0, 1)),
            ])
        return features_list

    pairs_w_k = [(5, 3), (5, 5), (5, 7), (7, 3),
                 (7, 5), (7, 7), (9, 3), (9, 5), (9, 7)]

    dataset_name = 'Indian_Pines'
    dataset = Dataset(dataset_name)
    p = dataset.dimension
    features_list = get_features(pairs_w_k, p)
    main(
        dataset=dataset,
        crop_image=False,
        nb_threads=os.cpu_count(),
        pairs_window_size_nb_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        nb_init=10,
        nb_iter_max=100
    )

    dataset_name = 'Pavia'
    dataset = Dataset(dataset_name)
    p = dataset.dimension
    features_list = get_features(pairs_w_k, p)
    main(
        dataset=dataset,
        crop_image=False,
        nb_threads=os.cpu_count(),
        pairs_window_size_nb_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        nb_init=10,
        nb_iter_max=100
    )

    dataset_name = 'Salinas'
    dataset = Dataset(dataset_name)
    p = dataset.dimension
    features_list = get_features(pairs_w_k, p)
    main(
        dataset=dataset,
        crop_image=False,
        nb_threads=os.cpu_count(),
        pairs_window_size_nb_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        nb_init=10,
        nb_iter_max=100
    )
