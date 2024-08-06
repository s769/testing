# petsc-test

build with `bash buildme.sh`- make sure any MPI and CUDA modules are loaded first
run with `mpiexec -np <num_procs> ./test [-N <array_size>]` or similar.

`N` is the global array length, but is stored only on the first process (default value is `49000`).
