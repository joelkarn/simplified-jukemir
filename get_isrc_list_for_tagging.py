import numpy as np
import argparse
import pathlib
import random
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files.")

    parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
    parser.add_argument("dimension", type=str, help="Which dimension (0-9) to investigate.")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)

    file_paths = sorted(list(input_dir.glob("**/*.ogg")))
    random.seed(0)
    random.shuffle(file_paths)

    y = np.array([os.path.split(p)[0].split('/')[-1].split('-')[int(args.dimension)] for p in file_paths])

    # Run cross-validation with confusion matrix evaluation
    unique, counts = np.unique(y, return_counts=True)
    min_counts = np.min(counts)
    print(min_counts)

    new_y = []
    for label in unique:
        indices = np.where(y == label)[0]
        new_y.append(y[indices[:min_counts]])

    new_y = np.concatenate(new_y)