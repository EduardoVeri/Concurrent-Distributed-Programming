import ctypes
import numpy as np
from .base_solution import BaseDiffusionEquation, DiffEqArgs
from ctypes import c_double, c_int, POINTER, byref


class CUDADiffusionEquation(BaseDiffusionEquation):
    """
    CUDA implementation of the diffusion equation solver.
    Uses a context manager to ensure CUDA resources are properly managed.
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
        self._cuda_initialized = False
        self._define_c_functions()
        self._cuda_init()

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context and ensure CUDA resources are freed.
        """
        self.finalize()

    def __del__(self):
        """
        Destructor to ensure CUDA resources are freed if not already done.
        """
        self.finalize()

    def _define_c_functions(self):
        """
        Define the argument and return types for the CUDA-related C functions.
        """
        required_functions = ['cuda_init', 'cuda_diff_eq', 'cuda_get_result', 'cuda_finalize']
        for func in required_functions:
            if not hasattr(self.lib, func):
                raise AttributeError(f"The shared library does not have '{func}' function.")

        # Define cuda_init
        self.lib.cuda_init.argtypes = [
            POINTER(c_double), POINTER(c_double), POINTER(DiffEqArgs)
        ]
        self.lib.cuda_init.restype = None

        # Define cuda_step
        self.lib.cuda_diff_eq.argtypes = [POINTER(DiffEqArgs)]
        self.lib.cuda_diff_eq.restype = c_double

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
        if self._cuda_initialized:
            return  # Avoid re-initialization

        N = self.args.N
        size = N * N

        # Flatten the concentration matrices
        self._h_C_flat = self.C.flatten().astype(np.float64)
        self._h_C_new_flat = self.C_new.flatten().astype(np.float64)

        # Convert numpy arrays to ctypes pointers
        h_C_flat_ptr = self._h_C_flat.ctypes.data_as(POINTER(c_double))
        h_C_new_flat_ptr = self._h_C_new_flat.ctypes.data_as(POINTER(c_double))

        # Call cuda_init
        self.lib.cuda_init(h_C_flat_ptr, h_C_new_flat_ptr, byref(self.args))
        self._cuda_initialized = True

    def reset_concentration_matrix(self, N: int):
        """
        Override to reset device memory if concentration matrices are reset.
        """
        super().reset_concentration_matrix(N)
        self._cuda_initialized = False
        self._cuda_init()

    def step(self) -> float:
        """
        Perform a single CUDA-parallel step of the diffusion equation solver.
        """
        if not self._cuda_initialized:
            self._cuda_init()

        diff = self.lib.cuda_diff_eq(byref(self.args))
        # No need to swap in Python; swapping is handled in CUDA code
        return diff

    def get_result(self):
        """
        Retrieve the concentration matrix from device to host.
        """
        N = self.args.N
        h_C_flat_ptr = self._h_C_flat.ctypes.data_as(POINTER(c_double))
        self.lib.cuda_get_result(h_C_flat_ptr, N)
        self.C = self._h_C_flat.reshape((N, N))

    def finalize(self):
        """
        Free device memory if not already freed.
        """
        if self._cuda_initialized:
            self.lib.cuda_finalize()
            self._cuda_initialized = False


if __name__ == "__main__":
    library_path = "build/libcuda.so"

    with CUDADiffusionEquation(
        library_path,
        N=500,
        D=0.1,
        DELTA_T=0.01,
        DELTA_X=1.0,
        initial_concentration_points={(250, 250): 1.0},
    ) as diffusion:
        for _ in range(1000):
            diffusion.step()
        diffusion.get_result()