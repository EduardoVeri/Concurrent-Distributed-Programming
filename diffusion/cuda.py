# cuda_solution.py
import ctypes
import numpy as np
from .base_solution import BaseDiffusionEquation, DiffEqArgs
from ctypes import c_double, c_int, POINTER, byref


class CUDADiffusionEquation(BaseDiffusionEquation):
    """
    CUDA implementation of the diffusion equation solver.
    """

    def __init__(
        self,
        library_path,
        initial_concentration_points=None,
        N=100,
        D=0.1,
        DELTA_T=0.01,
        DELTA_X=1.0,
    ):
        super().__init__(
            library_path, initial_concentration_points, N, D, DELTA_T, DELTA_X
        )
        self._define_c_functions()
        self._cuda_init_called = False
        self._cuda_init()

    def _define_c_functions(self):
        """
        Define the argument and return types for the CUDA-related C functions.
        """
        required_functions = ['cuda_init', 'cuda_step', 'cuda_get_result', 'cuda_finalize']
        for func in required_functions:
            if not hasattr(self.lib, func):
                raise AttributeError(f"The shared library does not have '{func}' function.")

        # Define cuda_init
        self.lib.cuda_init.argtypes = [
            POINTER(c_double), POINTER(c_double), POINTER(DiffEqArgs)
        ]
        self.lib.cuda_init.restype = None

        # Define cuda_step
        self.lib.cuda_step.argtypes = [POINTER(DiffEqArgs)]
        self.lib.cuda_step.restype = c_double

        # Define cuda_get_result
        self.lib.cuda_get_result.argtypes = [POINTER(c_double), c_int]
        self.lib.cuda_get_result.restype = None

        # Define cuda_finalize
        self.lib.cuda_finalize.argtypes = []
        self.lib.cuda_finalize.restype = None

    def _cuda_init(self):
        """
        Initialize device memory and copy data from host to device.
        """
        if self._cuda_init_called:
            return  # Avoid re-initialization

        N = self.args.N
        size = N * N

        # Flatten the concentration matrices
        h_C_flat = self.C.flatten().astype(np.float64)
        h_C_new_flat = self.C_new.flatten().astype(np.float64)

        # Convert numpy arrays to ctypes pointers
        h_C_flat_ptr = h_C_flat.ctypes.data_as(POINTER(c_double))
        h_C_new_flat_ptr = h_C_new_flat.ctypes.data_as(POINTER(c_double))

        # Call cuda_init
        self.lib.cuda_init(h_C_flat_ptr, h_C_new_flat_ptr, byref(self.args))
        self._cuda_init_called = True

    def reset_concentration_matrix(self, N: int):
        """
        Override to reset device memory if concentration matrices are reset.
        """
        super().reset_concentration_matrix(N)
        self._cuda_init_called = False
        self._cuda_init()

    def step(self) -> float:
        """
        Perform a single CUDA-parallel step of the diffusion equation solver.
        """
        diff = self.lib.cuda_step(byref(self.args))
        # No need to swap in Python; swapping is handled in CUDA code
        return diff  # Return value can be modified based on your needs

    def get_result(self):
        """
        Retrieve the concentration matrix from device to host.
        """
        N = self.args.N
        size = N * N
        h_C_flat = np.zeros(size, dtype=np.float64)
        h_C_flat_ptr = h_C_flat.ctypes.data_as(POINTER(c_double))
        self.lib.cuda_get_result(h_C_flat_ptr, N)
        self.C = h_C_flat.reshape((N, N))

    def finalize(self):
        """
        Free device memory.
        """
        self.lib.cuda_finalize()
        self._cuda_init_called = False