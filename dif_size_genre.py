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
    # random.seed(0)
    # random.shuffle(npy_paths)

    # Loads data
    X = np.array([np.load(p) for p in npy_paths])
    y = np.array([os.path.split(p)[0].split('/')[-1] for p in npy_paths])

    # Run cross-validation with confusion matrix evaluation
    unique, counts = np.unique(y, return_counts=True)

    data_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    scores_list = [0]*len(data_sizes)
    # make new X and y with equal number of samples per class
    i = 0
    for data_size in data_sizes:
        random.seed(i+2)

        # shuffle X and y and get data_size samples from each class
        new_X = []
        new_y = []
        for label in unique:
            indices = np.where(y == label)[0]
            random.shuffle(indices)
            new_X.append(X[indices[:data_size]])
            new_y.append(y[indices[:data_size]])

        new_X = np.concatenate(new_X)
        new_y = np.concatenate(new_y)

        clf = make_pipeline(StandardScaler(), SVC())
        if data_size == 5:
            scores = cross_val_score(clf, new_X, new_y, cv=5, scoring='f1_macro')

        else:
            scores = cross_val_score(clf, new_X, new_y, cv=10, scoring='f1_macro')
        scores_list[i] = np.mean(scores)
        i += 1
    print(scores_list)
