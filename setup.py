from setuptools import setup, find_packages

setup(
    name='pyCovariance',
    version='0.0',
    description='Covariance estimation and clustering using Riemannian geometry',
    url='https://github.com/antoinecollas/pyCovariance',
    author='Antoine Collas',
    keywords='Riemann, Riemannian, geometry, clustering, estimation, signal, processing, hyperspectral, SAR',
    packages=find_packages(),
    python_requires='>=3.6, <4',
    install_requires=['autograd', 'matplotlib', 'numpy', 'pymanopt', 'scikit-learn', 'scipy', 'seaborn', 'tqdm' ],
)

