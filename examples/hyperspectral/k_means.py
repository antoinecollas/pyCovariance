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

    # EVALUATION ACCORDING TO WINDOW_SIZE AND NB_BANDS

    hp = HyperparametersKMeans(
        crop_image=crop_image,
        nb_threads=nb_threads,
        pca=None,
        nb_bands_to_select=None,
        mask=mask,
        window_size=None,
        feature=None,
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

        hp.window_size = w_size
        hp.nb_bands_to_select = p

        features = features_list[i]

        prefix = 'w' + str(w_size) + '_p' + str(p)
        folder = os.path.join(folder_main, prefix)

        # K means and evaluations
        mIoUs = list()
        OAs = list()
        features_str = list()

        for i, feature in enumerate(features):
            hp.feature = feature

            print()

            pattern = re.compile(r'tau_\w*_UUH_\w*')
            if pattern.match(str(hp.feature)):
                hp.pca = False
            else:
                hp.pca = True

            print('Feature:', str(hp.feature))

            C, criterion_values = K_means_hyperspectral_image(dataset, hp)
            h = w = (max_w_size-w_size)//2
            if (h > 0) and (w > 0):
                C = C[h:-h, w:-w]

            prefix_f_name = str(i)
            mIoU, OA = evaluate_and_save_clustering(C, dataset,
                                                    hp, folder, prefix_f_name)
            mIoUs.append(mIoU)
            OAs.append(OA)
            features_str.append(str(hp.feature))

            # save plot of within classes variances
            fig, ax = plt.subplots(1)
            for c_value in criterion_values:
                x = list(range(len(c_value)))
                plt.plot(x, c_value, '+--')
            plt.ylabel('sum of within-classes variances')
            plt.title('Criterion values of ' + str(hp.feature) + ' feature.')
            temp = 'criterion_' + prefix + '_' + str(hp.feature)
            path = os.path.join(folder, temp)
            plt.savefig(path)

        # Bar plot of mIoUs
        fig, ax = plt.subplots(1)
        ax.bar(features_str, mIoUs, align='center')
        ax.set_ylim(0, 1)
        plt.ylabel('mIoU')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        path = os.path.join(folder, 'mIoU_'+prefix)
        plt.savefig(path)

        # Bar plot of OAs
        fig, ax = plt.subplots(1)
        ax.bar(features_str, OAs, align='center')
        ax.set_ylim(0, 1)
        plt.ylabel('OA')
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        path = os.path.join(folder, 'OA_'+prefix)
        plt.savefig(path)


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
                tau_UUH(w*w, p, k, weights=(1/(w*w), 1/k)),
                tau_UUH(w*w, p, k, weights=(0, 1)),
            ])
        return features_list

    pairs_w_k = [(5, 3), (5, 5), (5, 7), (7, 3), (7, 5), (7, 7), (9, 3), (9, 5), (9, 7)]

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
