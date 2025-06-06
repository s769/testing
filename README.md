## petsc-test

build with `bash buildme.sh`- make sure any MPI and CUDA modules are loaded first
run with `mpiexec -np <num_procs> ./test [-N <array_size>]` or similar.

`N` is the global array length, but is stored only on the first process (default value is `49000`).



## SBGEMV comparison tests

Build cuBLAS version with:

```
nvcc -std=c++17 -o sbgemv_benchmark_cu sbgemv_cu.cpp -lcublas -O3
```

Build rocBLAS version with:

```
hipcc -o sbgemv_benchmark_roc sbgemv_roc.cpp -lrocblas -O3
```

Run with:

```
./sbgemv_benchmark_<cu/roc> <trans> <m> <n> <batch_count> <datatype> [verify]
```

where:
- `trans` is
    - `N` for no op
    - `T` for transpose
    - `H` for conjugate transpose
- The matrices are `m x n`, and there are `batch_count` of them
- `datatype` is
    - `s` for float
    - `d` for double
    - `c` for float complex
    - `z` for double complex
- If `verify` is passed, it will check the result on CPU (this could be slow for large sizes)
