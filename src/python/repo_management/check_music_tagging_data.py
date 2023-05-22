import argparse
import pathlib
import numpy as np
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the count of different tags.")
    parser.add_argument("input_dir", type=str, help="Path to music tagging parent audio directory.")
    parser.add_argument("dimension", type=str, help="Which dimension (0-9) to investigate.")
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)
    input_paths = sorted(list(input_dir.glob("**/*.ogg")))
    dim = args.dimension
    y = np.array([os.path.split(p)[0].split('/')[-1].split('-')[int(dim)] for p in input_paths])
    unique, counts = np.unique(y, return_counts=True)

    with open('real_definite_paths.txt', 'r') as f:
        already_exctracted = [pathlib.Path(line.strip()) for line in f]

    print(unique, counts)
    # get a list of
