cmake_minimum_required(VERSION 3.12)
project(Diffusion-Equation)

# ------------------------------------------------------------------
# Options for Verbose and Evaluation Modes
# ------------------------------------------------------------------
option(ENABLE_VERBOSE "Build in verbose mode (defines VERBOSE)" OFF)
option(ENABLE_EVALUATE "Build in evaluating mode (defines EVALUATE)" OFF)

if (ENABLE_VERBOSE)
    add_compile_definitions(VERBOSE)
    message(STATUS "Verbose mode enabled: VERBOSE defined")
endif()

if (ENABLE_EVALUATE)
    add_compile_definitions(EVALUATE)
    message(STATUS "Evaluation mode enabled: EVALUATE defined")
endif()

# Option to disable compiler optimizations
option(NO_OPTIMIZATION "Disable compiler optimizations" OFF)

if (NO_OPTIMIZATION)
    message(STATUS "Compiler optimizations are disabled.")
    # Remove existing optimization flags and set to -O0
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_C_FLAGS_RELEASE "") # Optionally clear release flags
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "") # Clear RelWithDebInfo flags if necessary
endif()

# Specify the include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ----------------------------
# Find OpenMP
# ----------------------------
find_package(OpenMP REQUIRED)

if (NOT OpenMP_C_FOUND)
    message(FATAL_ERROR "OpenMP not found")
endif()

# ----------------------------
# Static Library for Utils
# ----------------------------
add_library(utils STATIC src/utils.c)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# ----------------------------
# Shared Library
# ----------------------------
set(LIB_SOURCES
    src/sequential.c
    src/omp.c
)

find_package(CUDAToolkit) # Find CUDA

if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found. Enabling CUDA support.")
    set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75)
    list(APPEND LIB_SOURCES src/cuda.cu)
else()
    message(STATUS "CUDA not found. Disabling CUDA support.")
endif()

find_package(MPI) # Find MPI

if (MPI_C_FOUND)
    message(STATUS "MPI found. Enabling MPI support.")
    include_directories(${MPI_C_INCLUDE_PATH})
    list(APPEND LIB_SOURCES src/mpi.c)
else()
    message(STATUS "MPI not found. Disabling MPI support.")
endif()

# Create the shared library and add BUILD_SHARED definition
add_library(DiffusionEquation SHARED ${LIB_SOURCES})
target_compile_definitions(DiffusionEquation PRIVATE BUILD_SHARED)
target_link_libraries(DiffusionEquation PRIVATE utils OpenMP::OpenMP_C)
target_include_directories(DiffusionEquation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# ----------------------------
# Standalone Executables
# ----------------------------
function(create_executable exec_name src_file)
    add_executable(${exec_name} ${src_file})
    target_link_libraries(${exec_name} PRIVATE utils OpenMP::OpenMP_C)
endfunction()

create_executable(sequential src/sequential.c)
create_executable(omp src/omp.c)

if (CUDAToolkit_FOUND)
    create_executable(cuda src/cuda.cu)
endif()

if (MPI_C_FOUND)
    create_executable(mpi src/mpi.c)
    target_link_libraries(mpi PRIVATE ${MPI_C_LIBRARIES})
endif()
