import time
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from diffusion import SequentialDiffusionEquation, OMPdiffusionEquation, CUDADiffusionEquation
import pandas as pd
import numpy as np
import argparse


def standard_deviation(arr: list) -> float:
    mean = sum(arr) / len(arr)
    return (sum((x - mean) ** 2 for x in arr) / len(arr)) ** 0.5


def get_args():
    parser = argparse.ArgumentParser(description="Run the diffusion simulation.")
    parser.add_argument(
        "-n",
        type=int,
        default=2000,
        help="Size of the concentration matrix (NxN).",
    )
    parser.add_argument(
        "-d",
        type=float,
        default=0.1,
        help="Diffusion coefficient.",
    )
    parser.add_argument(
        "-dt",
        type=float,
        default=0.01,
        help="Time step.",
    )
    parser.add_argument(
        "-dx",
        type=float,
        default=1.0,
        help="Spatial step.",
    )
    parser.add_argument(
        "--num_iterations",
        "-ni",
        type=int,
        default=500,
        help="Number of iterations to run the simulation.",
    )
    parser.add_argument(
        "--num_repeats",
        "-nr",
        type=int,
        default=10,
        help="Number of times to repeat the simulation.",
    )
    return parser.parse_args()

def main():
    

if __name__ == "__main__":
    args = get_args()

    for solution_type in [SequentialDiffusionEquation, OMPdiffusionEquation, CUDADiffusionEquation]:
        
        
        for _ in range(args.num_repeats):
            with solution_type() as diffusion:
                diffusion.reset_concentration_matrix(args.N)
                diffusion.set_diffusion_parameters(args.D, args.DELTA_T, args.DELTA_X)
                diffusion.set_initial_concentration()

                start_time = time.time()
                for _ in range(args.num_iterations):
                    diffusion.step()
                end_time = time.time()
                
                
    