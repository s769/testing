git pull
cmake . -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_COMPILER=nvcc  -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CXX_FLAGS="-O3" -DCMAKE_CUDA_FLAGS="-ccbin mpicxx" -DCMAKE_INCLUDE_PATH=/work/08435/srvenkat/ls6/petsc/arch-linux-c-debug/include -DCMAKE_LIBRARY_PATH=/work/08435/srvenkat/ls6/petsc/arch-linux-c-debug/lib
make -j