from .base_solution import BaseDiffusionEquation
import ctypes
from ctypes import c_double


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
