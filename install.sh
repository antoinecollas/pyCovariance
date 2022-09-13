# create conda environment
conda create -n pyCovariance python=3.8 --yes 
eval "$(conda shell.bash hook)"  # bug fix: https://github.com/conda/conda/issues/7980#issuecomment-492784093 
conda activate pyCovariance

# install libraries
pip install -r requirements.txt

# install pymanopt separately
# because of https://github.com/pymanopt/pymanopt/issues/161
pip install git+https://github.com/antoinecollas/pymanopt@master

# install pyCovariance
python setup.py install
