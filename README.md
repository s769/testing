# petsc-test

build with `bash buildme.sh`- make sure any MPI and CUDA modules are loaded first
run with `mpiexec -np 3 ./test -proc_rows 3 -proc_cols 1 -n 10 -use_gpu_aware_mpi 0` or similar

`proc_rows` and `proc_cols` control the size of a 2D processor grid. `n` is the global array length, which is stored on the first row of processors.
