
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(test_petsc LANGUAGES CXX CUDA C)
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

add_executable(test_petsc ${SOURCE_FILES})


set_target_properties( test_petsc
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
target_link_options(test_petsc PUBLIC)

target_link_libraries(${PETSC_LIBRARIES})

