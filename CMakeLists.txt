
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(test LANGUAGES CXX CUDA C)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")

include(FindCUDA)
FIND_PACKAGE(PETSc REQUIRED)







file(GLOB_RECURSE INC_FILES shared.cuh)
file(GLOB_RECURSE SRC_FILES main.cu)



INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PETSC_INCLUDES})
LINK_DIRECTORIES(${PETSC_LIBRARIES} ${CUDA_LIBRARIES})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})

set(SOURCE_FILES ${INC_FILES} ${SRC_FILES})

add_executable(test ${SOURCE_FILES})


set_target_properties(test
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
target_link_options(test PUBLIC)

target_link_libraries(test ${PETSC_LIBRARIES})

# cmake . -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_COMPILER=nvcc  -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CXX_FLAGS="-O3" -DCMAKE_CUDA_FLAGS="-ccbin mpicxx" -DCMAKE_INCLUDE_PATH=/work/08435/srvenkat/ls6/petsc/arch-linux-c-debug/include -DCMAKE_LIBRARY_PATH=/work/08435/srvenkat/ls6/petsc/arch-linux-c-debug/lib
