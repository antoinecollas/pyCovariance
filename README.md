# pyCovariance

![Build](https://github.com/antoinecollas/pyCovariance/workflows/pyCovariance-package/badge.svg)

pyCovariance is a python package for statistical estimation and clustering/classification on Riemannian manifolds.

It implements the following pipeline (where M is a Riemannian manifold):

![pipeline](https://github.com/antoinecollas/pyCovariance/blob/master/pipeline.png?raw=true)

This pipeline can be applied to many other types of data than images such as time-series.

## Installation

The script `install.sh` creates a conda environment with everything needed to run the examples of this repo and installs the package:

```
./install.sh
```

## Check

To check the installation, activate the created conda environment `pyCovariance` and run the unit tests:

```
conda activate pyCovariance
nose2 -v --with-coverage && coverage combine
```


## Run examples

To run examples, run the scripts from the folder `examples/` e.g.

```
python examples/hyperspectral/demo.py
```


## Cite

If you use this code please cite one of the following papers:

```
@misc{collas22MSG,
      author = {Collas, Antoine and Breloy, Arnaud and Ren, Chengfang and Ginolhac, Guillaume and Ovarlez, Jean-Philippe},
      title = {Riemannian optimization for non-centered mixture of scaled Gaussian distributions},
      year = {2022},
      url = {https://arxiv.org/abs/2209.03315}
}
```

```
@ARTICLE{collas2021ppca,
      author = {Collas, Antoine and Bouchard, Florent and Breloy, Arnaud and Ginolhac, Guillaume and Ren, Chengfang and Ovarlez, Jean-Philippe},
      journal = {IEEE Transactions on Signal Processing}, 
      title = {Probabilistic PCA From Heteroscedastic Signals: Geometric Framework and Application to Clustering}, 
      year = {2021},
      volume = {69},
      number = {},
      pages = {6546-6560},
      doi = {10.1109/TSP.2021.3130997}
}
```

```
@INPROCEEDINGS{collas21tylertype,
      author={Collas, A. and Bouchard, F. and Breloy, A. and Ren, C. and Ginolhac, G. and Ovarlez, J.-P.},
      booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      title={A Tyler-Type Estimator of Location and Scatter Leveraging Riemannian Optimization},
      year={2021},
      volume={},
      number={},
      pages={5160-5164},
      doi={10.1109/ICASSP39728.2021.9414974}
}
```
