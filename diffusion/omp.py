from .base_solution import BaseDiffusionEquation
import ctypes
from ctypes import c_double, c_int

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