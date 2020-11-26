from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import os

from pyCovariance.features import\
        covariance,\
        intensity_euclidean,\
        mean_pixel_euclidean,\
        pixel_euclidean

from hyperspectral_functions import\
        evaluate_and_save_clustering,\
        HyperparametersKMeans,\
        K_means_hyperspectral_image

matplotlib.use('Agg')


def main():
    dataset_name = 'Indian_Pines'

    # folder path to save files
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    folder = os.path.join('results', dataset_name, date_str)

    # EVALUATION ACCORDING TO WINDOW_SIZE AND NB_BANDS

    hp = HyperparametersKMeans(
        crop_image=False,
        enable_multi=True,
        pca=True,
        nb_bands_to_select=None,
        mask=True,
        window_size=None,
        features=None,
        nb_init=10,
        nb_iter_max=100,
        eps=1e-3
    )

    pairs_w_p = [(3, 4), (5, 4), (5, 10), (7, 4), (7, 10), (7, 20),
                 (9, 4), (9, 10), (9, 20), (9, 40)]

    for w, p in pairs_w_p:
        print('w=', w, 'p=', p)

        hp.nb_bands_to_select = p
        hp.window_size = w

        features_list = [
            intensity_euclidean(),
            'sklearn',
            pixel_euclidean(p),
            mean_pixel_euclidean(p),
            covariance(p),
        ]

        prefix = 'w' + str(w) + '_p' + str(p)

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
            prefix_f_name = str(i) + '_' + prefix
            mIoU, OA = evaluate_and_save_clustering(C, dataset_name,
                                                    hp, folder, prefix_f_name)
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
