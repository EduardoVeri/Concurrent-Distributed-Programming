
## Build the Project
This project is built using CMake. To build the project, you need to have CMake installed on your system. If you don't have CMake installed, you can install it using the following command:

```bash
sudo apt-get install cmake build-essential -y
```

To build the project, you can use the following commands:
```bash
mkdir build
cmake -B build -S . [-DNO_OPTIMIZATION=ON/OFF]
cmake --build build
```

## Using the C executable
The C executable provides a command-line interface for solving diffusion equations. The executable takes the following command-line arguments:

- `N`: Grid size
- `T`: Total iterations
- `D`: Diffusion coefficient
- `dt`: Time step
- `dx`: Spatial step
- `omp`: Number of OpenMP threads

```bash
./build/sequential <N> <T> <D> <dt> <dx>
./build/omp <N> <T> <D> <dt> <dx> <omp>
```

Here is an example of how to run the C executable:
```bash
time ./build/sequential 100 1000 0.1 0.01 1.0
time ./build/omp 100 1000 0.1 0.01 1.0 4
```

## Using the Python Module
The diffusion Python module provides an interface for solving diffusion equations using the shared C library compiled with the CMake. This allows for efficient numerical simulations by leveraging the computational speed of C while maintaining the flexibility and ease of use of Python.

This will be useful for users who are not familiar with C programming or who want to use the diffusion solver in a Python environment to create graphical visualizations or to integrate it with other Python libraries.

Before using the Python module, ensure that you have built the shared C library (libDiffusionEquation.so) as per the build instructions above.

To install the Python module, run:
```bash
pip install -e .
```

The `-e` flag installs the module in editable mode, which means that any changes made to the source code will be reflected in the installed module without needing to reinstall it.

### Usage Example
Here is an example of how to use the diffusion Python module to solve a 2D diffusion equation:

```python
from diffusion import DiffusionEquation

# Path to the compiled shared library
library_path = "build/libDiffusionEquation.so"

# Initialize the DiffusionEquation class
diffusion = DiffusionEquation(
    library_path=library_path,
    N=500,             # Grid size
    D=0.1,             # Diffusion coefficient
    DELTA_T=0.01,      # Time step
    DELTA_X=1.0,       # Spatial step
)

# Set initial concentration (optional)
initial_points = {(250, 250): 1.0}  # Set concentration at the center
diffusion.set_initial_concentration(initial_points)

# Perform diffusion steps
for _ in range(1000):
    diffusion.sequential_step()  # Use omp_step() for OpenMP parallel computation
    
# Access the concentration matrix
concentration = diffusion.concentration_matrix
print("Concentration at center:", concentration[diffusion.N // 2, diffusion.N // 2])
```
