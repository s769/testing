# petsc-test

build with `bash buildme.sh`- make sure any MPI and CUDA modules are loaded first
run with `mpiexec -np 3 ./test [-N]` or similar

`N` is the global array length, but is stored only on the first process.
