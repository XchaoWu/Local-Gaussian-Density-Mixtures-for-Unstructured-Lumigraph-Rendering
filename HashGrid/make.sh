export CUDA_HOME=/usr/local/cuda
rm -rf build 
python setup.py build 
cp build/lib.*/HASHGRID.*.so lib/HASHGRID.so