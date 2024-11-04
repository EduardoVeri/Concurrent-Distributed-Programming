import ctypes
import numpy as np
import os
from ctypes import POINTER, c_int, c_double, Structure


class DiffEqArgs(Structure):
    _fields_ = [
        ("N", c_int),
        ("D", c_double),
        ("DELTA_T", c_double),
        ("DELTA_X", c_double),
    ]


class DiffusionEquation:
    def __init__(
        self,
        library_path: str,
        initial_concentration_points_and_values: dict = None,
        N: int = 100,
        D: float = 1.0,
        DELTA_T: float = 0.01,
        DELTA_X: float = 1.0,
    ):
        self.lib = self._load_library(library_path)
        self._define_c_functions()

        # Set default values for diffusion parameters
        self.N = N
        self.D = D
        self.DELTA_T = DELTA_T
        self.DELTA_X = DELTA_X

        # Create a DiffEqArgs instance
        self.args = DiffEqArgs(
            N=self.N, D=self.D, DELTA_T=self.DELTA_T, DELTA_X=self.DELTA_X
        )

        # Create initial concentration matrices
        self.C = np.zeros((self.N, self.N), dtype=np.float64)
        
        if initial_concentration_points_and_values:
            for point, value in initial_concentration_points_and_values.items():
                self.C[point[0], point[1]] = value
        else:
            self.C[self.N // 2, self.N // 2] = 1.0
        
        self.C = self._convert_numpy_to_double_ptr_ptr(self.C)
        
        self.C_new = self._convert_numpy_to_double_ptr_ptr(
            np.zeros((self.N, self.N), dtype=np.float64)
        )
        print(self.C[self.N // 2][self.N // 2])

    def _load_library(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shared library not found at path: {path}")
        return ctypes.CDLL(path)

    def _define_c_functions(self):
        # Define the argument and return types for sequential_diff_eq
        self.lib.sequential_diff_eq.argtypes = [
            POINTER(POINTER(c_double)),  # double** C
            POINTER(POINTER(c_double)),  # double** C_new
            POINTER(DiffEqArgs),  # DiffEqArgs* args
        ]
        self.lib.sequential_diff_eq.restype = None  # void

    def _convert_numpy_to_double_ptr_ptr(self, array: np.ndarray):
        if not (
            isinstance(array, np.ndarray)
            and array.dtype == np.float64
            and array.ndim == 2
            and array.flags["C_CONTIGUOUS"]
        ):
            raise ValueError(
                "Input must be a 2D C-contiguous NumPy array of type float64."
            )

        N = array.shape[0]
        # Create an array of pointers to each row
        row_type = POINTER(c_double) * N
        row_ptrs = row_type()
        for i in range(N):
            row_ptrs[i] = array[i].ctypes.data_as(POINTER(c_double))
        return row_ptrs

    def step(self):
        # Call the C function
        self.lib.sequential_diff_eq(self.C, self.C_new, ctypes.byref(self.args))
        # Swap the pointers
        self.C, self.C_new = self.C_new, self.C
        print(self.C[self.N // 2][self.N // 2])


# Example Usage
if __name__ == "__main__":
    # Path to the compiled shared library
    # Update this path based on your environment
    library_path = "/home/eduardovf/documents/projects/pcd/Concurrent-Distributed-Programming/build/libmylibrary.so"

    # Initialize the DiffusionEquation class
    diffusion = DiffusionEquation(library_path)

    # Perform a diffusion step
    for i in range(1000):
        diffusion.step()

    # print(diffusion.C_new[diffusion.N // 2, diffusion.N // 2])
    