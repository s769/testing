git pull
cmake . 
make -j

# run with:
# ibrun -np 3 ./test -proc_rows 3 -proc_cols 1 -n 10 -use_gpu_aware_mpi 0