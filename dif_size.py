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
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)

    npy_paths = sorted(list(input_dir.glob("**/*.npy")))

    dimensions = range(10)

    # data_sizes = list(range(100, 751, 50))
    # data_sizes.append(787)
    # dim = 1

    data_sizes = list(range(100, 1001, 50))
    dim = 0

    scores_list = [0]*len(data_sizes)
    # Loads data
    X = np.array([np.load(p) for p in npy_paths])


    y = np.array([os.path.split(p)[0].split('/')[-1].split('-')[int(dim)] for p in npy_paths])

    # remove all samples with label 'neither'
    indices = np.where(y == 'neither')[0]
    X = np.delete(X, indices, axis=0)
    y = np.delete(y, indices, axis=0)

    unique, counts = np.unique(y, return_counts=True)
    i = 0
    for n_samples in data_sizes:
        # make new X and y with equal number of samples per class
        random.seed(i+13337)
        new_X = []
        new_y = []
        for label in unique:
            indices = np.where(y == label)[0]
            random.shuffle(indices)
            new_X.append(X[indices[:n_samples]])
            new_y.append(y[indices[:n_samples]])

        new_X = np.concatenate(new_X)
        new_y = np.concatenate(new_y)

        clf = make_pipeline(StandardScaler(), SVC())
        scores = cross_val_score(clf, new_X, new_y, cv=10, scoring='f1_macro')
        scores_list[i] += np.mean(scores)
        i += 1
    print(scores_list)
