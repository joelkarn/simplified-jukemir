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

    npy_paths = sorted(list(input_dir.glob("**/*.npy")))
    random.seed(0)
    random.shuffle(npy_paths)

    # Loads data
    X = np.array([np.load(p) for p in npy_paths])
    y = np.array([os.path.split(p)[0].split('/')[-1].split('-')[int(args.dimension)] for p in npy_paths])

    # Run cross-validation with confusion matrix evaluation
    unique, counts = np.unique(y, return_counts=True)
    min_counts = np.min(counts)
    print(min_counts)

    # make new X and y with equal number of samples per class
    new_X = []
    new_y = []
    for label in unique:
        indices = np.where(y == label)[0]
        new_X.append(X[indices[:min_counts]])
        new_y.append(y[indices[:min_counts]])

    new_X = np.concatenate(new_X)
    new_y = np.concatenate(new_y)

    clf = make_pipeline(StandardScaler(), SVC())
    scores = cross_val_score(clf, new_X, new_y, cv=10)
    y_pred = cross_val_predict(clf, new_X, new_y, cv=10)
    conf_mat = confusion_matrix(new_y, y_pred)

    print('{:.1f} +- {:.1f}'.format(np.mean(scores) * 100, np.std(scores) * 100))
    print(conf_mat)

    # Plot confusion matrix with labels
    labels = sorted(set(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                                  display_labels=labels)
    disp.plot()

    # Save plot as image file
    plot_name = 'data/conf_mat/{}/eq_dist_{:.1f}_{:.1f}_cv10.png'.format(args.dimension, np.mean(scores) * 100, np.std(scores) * 100)
    plt.savefig(plot_name)
