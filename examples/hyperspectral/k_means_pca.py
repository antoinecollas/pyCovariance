from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import os

from pyCovariance.features import\
        intensity_euclidean,\
        mean_pixel_euclidean,\
        pixel_euclidean

from hyperspectral_functions import\
        evaluate_and_save_clustering,\
        HyperparametersKMeans,\
        K_means_hyperspectral_image


def main():
    matplotlib.use('Agg')

    dataset_name = 'Indian_Pines'

    # folder path to save files
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    folder = os.path.join('results', dataset_name, date_str)

    # EVALUATION OF PCA

    hp = HyperparametersKMeans(
        crop_image=True,
        enable_multi=True,
        pca=None,
        nb_bands_to_select=10,
        mask=True,
        window_size=7,
        features=None,
        nb_init=1,
        nb_iter_max=10,
        eps=1e-3
    )

    features_list = [
        intensity_euclidean(),
        'sklearn',
        pixel_euclidean(hp.nb_bands_to_select),
        mean_pixel_euclidean(hp.nb_bands_to_select),
    ]

    for pca in [False, True]:
        hp.pca = pca
        if pca:
            prefix = 'pca'
        else:
            prefix = 'no_pca'

        # K means and evaluations
        mIoUs = list()
        OAs = list()
        features_str = list()
        for i, features in enumerate(features_list):
            hp.features = features
            print()
            print('Features:', str(hp.features))
            C = K_means_hyperspectral_image(dataset_name, hp)
            print()
            mIoU, OA = evaluate_and_save_clustering(
                C, dataset_name, hp, folder, str(i) + '_' + prefix)
            mIoUs.append(mIoU)
            OAs.append(OA)
            features_str.append(str(features))

        # Bar plot of mIoUs
        fig, ax = plt.subplots(1)
        ax.bar(features_str, mIoUs, align='center')
        ax.set_ylim(0, 0.5)
        plt.ylabel('mIoU')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.5)
        path = os.path.join('results', dataset_name, date_str, 'mIoU'+prefix)
        plt.savefig(path)

        # Bar plot of OAs
        fig, ax = plt.subplots(1)
        ax.bar(features_str, OAs, align='center')
        ax.set_ylim(0, 1)
        plt.ylabel('OA')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.5)
        path = os.path.join('results', dataset_name, date_str, 'OA'+prefix)
        plt.savefig(path)


if __name__ == '__main__':
    main()
