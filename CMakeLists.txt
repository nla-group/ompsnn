cmake_minimum_required(VERSION 3.10)
project(SNN)

# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP detected: ${OpenMP_CXX_FLAGS}")
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

# Find BLAS (CBLAS is part of BLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "BLAS detected: ${BLAS_LIBRARIES}")
endif()

# Create the executable
aux_source_directory(examples SOURCES)
add_executable(example ${SOURCES})

# Link CBLAS and OpenMP
target_include_directories(example PRIVATE include)
target_link_libraries(example PRIVATE ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)