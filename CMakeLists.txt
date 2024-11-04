# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(Concurrent-Distributed-Programming)

# Set the C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Specify the include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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
)

# Create the shared library
add_library(mylibrary SHARED ${LIB_SOURCES})

# Define BUILD_SHARED to exclude main functions
target_compile_definitions(mylibrary PRIVATE BUILD_SHARED)

# Link the shared library with the utils static library
target_link_libraries(mylibrary PRIVATE utils)

# Specify the library's include directories
target_include_directories(mylibrary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# ----------------------------
# Standalone Executables
# ----------------------------

# Function to create executables
function(create_executable exec_name src_file)
    add_executable(${exec_name} ${src_file})
    # Link each executable with the utils static library
    target_link_libraries(${exec_name} PRIVATE utils)
endfunction()

# Create executables for each source file
create_executable(sequential_exec src/sequential.c)

# ----------------------------
# Installation (Optional)
# ----------------------------

# # Specify installation rules
# install(TARGETS mylibrary sequential_exec parallel_exec main_program
#         RUNTIME DESTINATION bin
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib)

# install(DIRECTORY include/ DESTINATION include)
