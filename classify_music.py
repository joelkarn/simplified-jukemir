import glob
import os
import random
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='read input and produce some output')

    parser.add_argument('features_path', type=str,
                        help='Path to the features directory')

    args = parser.parse_args()

    # Find numpy paths (and randomize to remove label ordering)
    npy_paths = sorted(glob.glob(args.features_path + '/*.npy'))
    random.seed(0)
    random.shuffle(npy_paths)

    # Loads data
    X = np.array([np.load(p) for p in npy_paths])
    y = np.array([os.path.split(p)[1].split('.')[0] for p in npy_paths])

    print(X.shape, y.shape)

    # Run cross-validation with confusion matrix evaluation
    clf = make_pipeline(StandardScaler(), SVC())
    scores = cross_val_score(clf, X, y, cv=10)
    y_pred = cross_val_predict(clf, X, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)

    print('{:.1f} +- {:.1f}'.format(np.mean(scores) * 100, np.std(scores) * 100))
    print(conf_mat)

    # Plot confusion matrix with labels
    labels = sorted(set(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                                  display_labels=labels)
    disp.plot()

    # Save plot as image file
    plt.savefig('conf_mat.png')