export CUDA_HOME=/usr/local/cuda
rm -r build 
python setup.py build 
cp build/lib.*/CUDA_EXT.*.so lib/CUDA_EXT.so
