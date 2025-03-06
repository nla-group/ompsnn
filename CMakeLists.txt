cmake_minimum_required(VERSION 3.10)
project(SNN)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP detected: ${OpenMP_CXX_FLAGS}")
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "BLAS detected: ${BLAS_LIBRARIES}")
endif()

aux_source_directory(examples SOURCES)
add_executable(example ${SOURCES})

target_include_directories(example PRIVATE include)
target_link_libraries(example PRIVATE ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)
