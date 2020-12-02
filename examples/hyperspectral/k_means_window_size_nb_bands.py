from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import os

from pyCovariance.features import\
        intensity_euclidean,\
        mean_pixel_euclidean,\
        pixel_euclidean,\
        covariance,\
        covariance_euclidean,\
        covariance_texture,\
        tau_UUH

from .hyperspectral_functions import\
        Dataset,\
        evaluate_and_save_clustering,\
        HyperparametersKMeans,\
        K_means_hyperspectral_image


def main(
    dataset,
    crop_image,
    enable_multi,
    pairs_window_size_nb_bands,
    mask,
    features_list,
    nb_init,
    nb_iter_max
):
    matplotlib.use('Agg')

    # folder path to save files
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    folder = os.path.join('results', dataset.name, date_str)

    # EVALUATION ACCORDING TO WINDOW_SIZE AND NB_BANDS

    hp = HyperparametersKMeans(
        crop_image=crop_image,
        enable_multi=enable_multi,
        pca=None,
        nb_bands_to_select=None,
        mask=mask,
        window_size=None,
        features=None,
        nb_init=nb_init,
        nb_iter_max=nb_iter_max,
        eps=1e-3
    )

    pairs_w_p = pairs_window_size_nb_bands
    max_w_size = pairs_w_p[-1][0]

    for i, (w_size, p) in enumerate(pairs_w_p):
        print()
        print('########################################################')
        print('w_size =', w_size)
        print('p =', p)
        print('########################################################')

        hp.nb_bands_to_select = p
        hp.window_size = w_size

        features = features_list[i]

        prefix = 'w' + str(w_size) + '_p' + str(p)

        # K means and evaluations
        mIoUs = list()
        OAs = list()
        features_str = list()

        for i, feature in enumerate(features):
            hp.features = feature

            print()

            if str(hp.features) == 'tau_UUH_Riemannian':
                hp.pca = False
            else:
                hp.pca = True

            print('Feature:', str(hp.features))

            C = K_means_hyperspectral_image(dataset, hp)
            h = w = (max_w_size-w_size)//2
            if (h > 0) and (w > 0):
                C = C[h:-h, w:-w]

            prefix_f_name = str(i) + '_' + prefix
            mIoU, OA = evaluate_and_save_clustering(C, dataset,
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
        path = os.path.join('results', dataset.name, date_str, 'mIoU'+prefix)
        plt.savefig(path)

        # Bar plot of OAs
        fig, ax = plt.subplots(1)
        ax.bar(features_str, OAs, align='center')
        ax.set_ylim(0, 1)
        plt.ylabel('OA')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.5)
        path = os.path.join('results', dataset.name, date_str, 'OA'+prefix)
        plt.savefig(path)


if __name__ == '__main__':

    dataset_name = 'Indian_Pines'
    dataset = Dataset(dataset_name)
    p = dataset.dimension

    pairs_w_k = [(5, 4), (5, 10), (7, 4), (7, 10), (7, 20),
                 (9, 4), (9, 10), (9, 20), (9, 40)]

    features_list = list()
    for pair in pairs_w_k:
        w, k = pair
        features_list.append([
            intensity_euclidean(),
            'sklearn',
            pixel_euclidean(k),
            mean_pixel_euclidean(k),
            covariance_euclidean(k),
            covariance(k),
            covariance_texture(w*w, k),
            tau_UUH(p, k, w*w)
        ])

    main(
        dataset=dataset,
        crop_image=False,
        enable_multi=True,
        pairs_window_size_nb_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        nb_init=5,
        nb_iter_max=100
    )
