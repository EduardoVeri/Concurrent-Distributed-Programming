import ctypes
import numpy as np
from os import path as os_path
from ctypes import POINTER, c_int, c_double, Structure
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional


class DiffEqArgs(Structure):
    """
    C-compatible structure to hold diffusion equation parameters.
    """

    _fields_ = [
        ("N", c_int),
        ("D", c_double),
        ("DELTA_T", c_double),
        ("DELTA_X", c_double),
    ]


class BaseDiffusionEquation(ABC):
    """
    Abstract base class for diffusion equation solvers.
    Encapsulates common functionality for both sequential and OMP implementations.
    """

    FUNCTION_PARAMS = [
        POINTER(POINTER(c_double)),  # double** C
        POINTER(POINTER(c_double)),  # double** C_new
        POINTER(DiffEqArgs),  # DiffEqArgs* args
    ]

    def __init__(
        self,
        library_path: str,
        initial_concentration_points: Optional[Dict[Tuple[int, int], float]] = None,
        N: int = 100,
        D: float = 0.1,
        DELTA_T: float = 0.01,
        DELTA_X: float = 1.0,
    ):
        """
        Initialize the diffusion equation solver.

        :param library_path: Path to the shared C library.
        :param initial_concentration_points: Dictionary of initial concentration points.
        :param N: Size of the concentration matrix (NxN).
        :param D: Diffusion coefficient.
        :param DELTA_T: Time step.
        :param DELTA_X: Spatial step.
        """
        self.lib = self._load_library(library_path)
        self._define_c_functions()

        self.args = DiffEqArgs(N=N, D=D, DELTA_T=DELTA_T, DELTA_X=DELTA_X)

        self.set_diffusion_parameters(D, DELTA_T, DELTA_X)
        self.reset_concentration_matrix(N)
        self.set_initial_concentration(initial_concentration_points)

    @staticmethod
    def _load_library(path: str) -> ctypes.CDLL:
        """
        Load the shared C library.

        :param path: Path to the shared library.
        :return: Loaded CDLL object.
        :raises FileNotFoundError: If the library does not exist at the given path.
        """
        if not os_path.exists(path):
            raise FileNotFoundError(f"Shared library not found at path: {path}")
        return ctypes.CDLL(path)

    @abstractmethod
    def _define_c_functions(self):
        """
        Define the argument and return types for the C functions.
        Must be implemented by subclasses.
        """
        pass

    def reset_concentration_matrix(self, N: int):
        """
        Reset the concentration matrices to zero.

        :param N: Size of the concentration matrix (NxN).
        """
        self.N = N
        self.args.N = self.N
        self.C = np.zeros((self.N, self.N), dtype=np.float64)
        self._C_ptr = self._convert_numpy_to_double_ptr_ptr(self.C)

        self.C_new = np.zeros((self.N, self.N), dtype=np.float64)
        self._C_new_ptr = self._convert_numpy_to_double_ptr_ptr(self.C_new)

    def set_diffusion_parameters(self, D: float, DELTA_T: float, DELTA_X: float):
        """
        Set the diffusion parameters.

        :param D: Diffusion coefficient.
        :param DELTA_T: Time step.
        :param DELTA_X: Spatial step.
        """
        self.D = D
        self.DELTA_T = DELTA_T
        self.DELTA_X = DELTA_X
        self.args.D = self.D
        self.args.DELTA_T = self.DELTA_T
        self.args.DELTA_X = self.DELTA_X

    def set_initial_concentration(
        self, points_and_values: Optional[Dict[Tuple[int, int], float]] = None
    ):
        """
        Set the initial concentration in the matrix.

        :param points_and_values: Dictionary with keys as (x, y) tuples and values as concentrations.
        :raises ValueError: If any point is out of bounds.
        """
        if not points_and_values:
            # Set default initial concentration at the center
            self.C[self.N // 2, self.N // 2] = 1.0
            return

        for point, value in points_and_values.items():
            if 0 <= point[0] < self.N and 0 <= point[1] < self.N:
                self.C[point[0], point[1]] = value
            else:
                raise ValueError(
                    f"Invalid point: {point}. Point must be within the concentration matrix bounds (N={self.N})."
                )

    def _convert_numpy_to_double_ptr_ptr(self, array: np.ndarray) -> Any:
        """
        Convert a 2D NumPy array to a C-compatible double**.

        :param array: 2D NumPy array of type float64 and C-contiguous.
        :return: C-compatible double** pointer.
        :raises ValueError: If the input array does not meet the required conditions.
        """
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
        row_type = POINTER(c_double) * N
        row_ptrs = row_type()
        for i in range(N):
            row_ptrs[i] = array[i].ctypes.data_as(POINTER(c_double))
        return row_ptrs

    @property
    def concentration_matrix(self) -> np.ndarray:
        """
        Get the current concentration matrix.

        :return: Current concentration as a NumPy array.
        """
        return self.C

    @abstractmethod
    def step(self) -> float:
        """
        Perform a single step of the diffusion equation solver.

        :return: The computed diffusion value.
        """
        pass


class SequentialDiffusionEquation(BaseDiffusionEquation):
    """
    Sequential implementation of the diffusion equation solver.
    """

    def __init__(
        self,
        library_path,
        initial_concentration_points=None,
        N=100,
        D=0.1,
        DELTA_T=0.01,
        DELTA_X=1,
    ):
        super().__init__(
            library_path, initial_concentration_points, N, D, DELTA_T, DELTA_X
        )

    def _define_c_functions(self):
        """
        Define the argument and return types for the sequential_diff_eq C function.
        """
        if not hasattr(self.lib, "sequential_diff_eq"):
            raise AttributeError(
                "The shared library does not have 'sequential_diff_eq' function."
            )

        self.lib.sequential_diff_eq.argtypes = self.FUNCTION_PARAMS
        self.lib.sequential_diff_eq.restype = c_double

    def step(self) -> float:
        """
        Perform a single sequential step of the diffusion equation solver.

        :return: The computed diffusion value.
        """
        diff = self.lib.sequential_diff_eq(
            self._C_ptr, self._C_new_ptr, ctypes.byref(self.args)
        )
        # Swap the concentration matrices
        self.C, self.C_new = self.C_new, self.C
        self._C_ptr, self._C_new_ptr = self._C_new_ptr, self._C_ptr
        return diff


class OMPdiffusionEquation(BaseDiffusionEquation):
    """
    OpenMP (parallel) implementation of the diffusion equation solver.
    """

    def __init__(
        self,
        library_path,
        initial_concentration_points=None,
        N=100,
        D=0.1,
        DELTA_T=0.01,
        DELTA_X=1,
    ):
        super().__init__(
            library_path, initial_concentration_points, N, D, DELTA_T, DELTA_X
        )

    def _define_c_functions(self):
        """
        Define the argument and return types for the OMP-related C functions.
        """
        required_functions = ["omp_diff_eq", "omp_set_num_threads"]
        for func in required_functions:
            if not hasattr(self.lib, func):
                raise AttributeError(
                    f"The shared library does not have '{func}' function."
                )

        # Define omp_diff_eq
        self.lib.omp_diff_eq.argtypes = self.FUNCTION_PARAMS
        self.lib.omp_diff_eq.restype = c_double

        # Define omp_set_num_threads
        self.lib.omp_set_num_threads.argtypes = [c_int]
        self.lib.omp_set_num_threads.restype = None  # void

    def set_num_threads(self, num_threads: int):
        """
        Set the number of OpenMP threads.

        :param num_threads: Number of threads to use.
        :raises ValueError: If num_threads is not positive.
        """
        if num_threads <= 0:
            raise ValueError("Number of threads must be positive.")
        self.lib.omp_set_num_threads(num_threads)

    def step(self) -> float:
        """
        Perform a single OpenMP-parallel step of the diffusion equation solver.

        :return: The computed diffusion value.
        """
        diff = self.lib.omp_diff_eq(
            self._C_ptr, self._C_new_ptr, ctypes.byref(self.args)
        )
        # Swap the concentration matrices
        self.C, self.C_new = self.C_new, self.C
        self._C_ptr, self._C_new_ptr = self._C_new_ptr, self._C_ptr
        return diff


if __name__ == "__main__":
    # Paths to your shared C libraries
    lib_path = "../build/libDiffusionEquation.so"

    # Initialize the sequential solver
    seq_solver = SequentialDiffusionEquation(
        library_path=lib_path, N=200, D=0.05, DELTA_T=0.02, DELTA_X=1.0
    )

    # Set the initial concentration at the center
    seq_solver.set_initial_concentration({(100, 100): 1.0})

    diff_seq = seq_solver.step()  # Perform a simulation step
    print(f"Sequential diffusion value: {diff_seq}")

    # Initialize the OpenMP solver
    omp_solver = OMPdiffusionEquation(
        library_path=lib_path, N=200, D=0.05, DELTA_T=0.02, DELTA_X=1.0
    )

    seq_solver.set_initial_concentration({(100, 100): 1.0})

    # Set the number of threads for OpenMP
    omp_solver.set_num_threads(8)

    diff_omp = omp_solver.step()
    print(f"OpenMP diffusion value: {diff_omp}")

    # Access the current concentration matrix
    current_conc = omp_solver.concentration_matrix
