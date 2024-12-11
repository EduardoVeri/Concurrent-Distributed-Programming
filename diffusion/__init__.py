from .base_solution import BaseDiffusionEquation
from .sequential import SequentialDiffusionEquation
from .omp import OMPdiffusionEquation

__all__ = ["BaseDiffusionEquation", "SequentialDiffusionEquation", "OMPdiffusionEquation"]
