from setuptools import setup, find_packages

setup(
    name='pyCovariance',
    description='Covariance estimation and clustering using Riemannian geometry',
    url='https://github.com/antoinecollas/pyCovariance',
    author='Antoine Collas',
    keywords='Riemannian, geometry, estimation, clustering, classification, covariance, subspace',
    packages=find_packages(),
    python_requires='>=3.8, <4',
    install_requires=[
        'autograd',
        'matplotlib',
        'numpy',
        'pymanopt',
        'scikit-learn',
        'scipy',
        'tikzplotlib',
        'tqdm' 
    ]
)

