# create conda environment
conda create -n pyCovariance python=3.7.6 --yes 
eval "$(conda shell.bash hook)"  # bug fix: https://github.com/conda/conda/issues/7980#issuecomment-492784093 
conda activate pyCovariance

# install libraries
pip install -r requirements.txt

# install custom pymanopt
DIR="../pymanopt"
if [ ! -d "$DIR" ]
then
	cd ..
	git clone https://github.com/antoinecollas/pymanopt
	cd -
fi
cd ../pymanopt
python setup.py install

# install pyCovariance
cd -
python setup.py install
