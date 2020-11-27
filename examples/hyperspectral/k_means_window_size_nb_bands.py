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
        crop_image=True,
        enable_multi=True,
        pca=True,
        nb_bands_to_select=None,
        mask=True,
        window_size=None,
        features=None,
        nb_init=1,
        nb_iter_max=10,
        eps=1e-3
    )

    pairs_w_p = [(3, 4), (5, 4)]
    max_w_size = pairs_w_p[-1][0]

    for w_size, p in pairs_w_p:
        print()
        print('##############')
        print('w_size=', w_size, 'p=', p)
        print('##############')

        hp.nb_bands_to_select = p
        hp.window_size = w_size

        features_list = [
            intensity_euclidean(),
            'sklearn',
            pixel_euclidean(p),
            mean_pixel_euclidean(p),
        ]

        prefix = 'w' + str(w_size) + '_p' + str(p)

        # K means and evaluations
        mIoUs = list()
        OAs = list()
        features_str = list()
        for i, features in enumerate(features_list):
            hp.features = features
            print()
            print('Features:', str(hp.features))
            C = K_means_hyperspectral_image(dataset_name, hp)
            h = w = (max_w_size-w_size)//2
            if (h > 0) and (w > 0):
                C = C[h:-h, w:-w]
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
