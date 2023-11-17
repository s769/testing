git pull
cmake . -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_COMPILER=nvcc  -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_FLAGS="-ccbin mpicxx" 
make -j

# run with:
# ibrun -np 3 ./test -proc_rows 3 -proc_cols 1 -n 10 -use_gpu_aware_mpi 0