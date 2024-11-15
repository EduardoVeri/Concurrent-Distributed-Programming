# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(Diffusion-Equation)

# ----------------------------
# Add Option to Disable Optimizations
# ----------------------------
option(NO_OPTIMIZATION "Disable compiler optimizations" OFF)

if (NO_OPTIMIZATION)
    message(STATUS "Compiler optimizations are disabled.")
    # Remove existing optimization flags and set to -O0
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_C_FLAGS_RELEASE "") # Optionally clear release flags
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "") # Clear RelWithDebInfo flags if necessary
endif()

# Set the C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

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

# Specify the include directories for utils
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# ----------------------------
# Shared Library
# ----------------------------

# Define the source files for the shared library
set(LIB_SOURCES
    src/sequential.c
    src/omp.c
)

# Create the shared library
add_library(DiffusionEquation SHARED ${LIB_SOURCES})

# Define BUILD_SHARED to exclude main functions
target_compile_definitions(DiffusionEquation PRIVATE BUILD_SHARED)

# Link the shared library with the utils static library and OpenMP
target_link_libraries(DiffusionEquation PRIVATE utils OpenMP::OpenMP_C)

# Specify the library's include directories
target_include_directories(DiffusionEquation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# ----------------------------
# Standalone Executables
# ----------------------------

# Function to create executables
function(create_executable exec_name src_file)
    add_executable(${exec_name} ${src_file})
    # Link each executable with the utils static library and OpenMP
    target_link_libraries(${exec_name} PRIVATE utils OpenMP::OpenMP_C)
endfunction()

# Create executables for each source file
create_executable(sequential src/sequential.c)
create_executable(omp src/omp.c)
