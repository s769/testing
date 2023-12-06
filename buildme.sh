git pull
cmake . -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_FLAGS="-ccbin mpicxx"
make -j

# run with:
# mpiexec -np 12 ./test -proc_rows 4 -proc_cols 3 -nm 3 -nt 12 -use_gpu_aware_mpi 0
