import pytest
import numpy as np
import sys
from diffusion import DiffusionEquation
import os


@pytest.fixture(scope="module")
def diffusion():
    library_path = "./build/libDiffusionEquation.so"
    # Ensure the shared library exists
    assert os.path.exists(library_path), f"Shared library not found at path: {library_path}"
    return DiffusionEquation(library_path)

def test_initial_concentration_default(diffusion):
    expected_value = 1.0
    actual_value = diffusion.concentration_matrix[diffusion.N // 2, diffusion.N // 2]
    assert actual_value == expected_value

def test_initial_concentration_custom():
    library_path = "./build/libDiffusionEquation.so"
    assert os.path.exists(library_path), f"Shared library not found at path: {library_path}"
    initial_points = {(10, 10): 0.5, (20, 20): 0.8}
    diffusion = DiffusionEquation(library_path, initial_concentration_points=initial_points)
    assert diffusion.concentration_matrix[10, 10] == 0.5
    assert diffusion.concentration_matrix[20, 20] == 0.8

def test_reset_concentration_matrix(diffusion):
    diffusion.reset_concentration_matrix(50)
    assert diffusion.N == 50
    assert np.array_equal(diffusion.concentration_matrix, np.zeros((50, 50)))

def test_set_diffusion_parameters(diffusion):
    diffusion.set_diffusion_parameters(0.2, 0.02, 2.0)
    assert diffusion.D == 0.2
    assert diffusion.DELTA_T == 0.02
    assert diffusion.DELTA_X == 2.0

def test_sequential_step(diffusion):
    initial_value = diffusion.concentration_matrix[diffusion.N // 2, diffusion.N // 2]
    diffusion.sequential_step()
    new_value = diffusion.concentration_matrix[diffusion.N // 2, diffusion.N // 2]
    assert initial_value != new_value

def test_set_num_threads(diffusion):
    diffusion.set_num_threads(4)
    # No direct way to test this; ensure no exceptions are raised