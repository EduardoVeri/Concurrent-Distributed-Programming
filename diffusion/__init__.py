from .base_solution import BaseDiffusionEquation
from .sequential import SequentialDiffusionEquation
from .omp import OMPdiffusionEquation
from .cuda import CUDADiffusionEquation

__all__ = [
    "BaseDiffusionEquation",
    "SequentialDiffusionEquation",
    "OMPdiffusionEquation",
    "CUDADiffusionEquation",
]
