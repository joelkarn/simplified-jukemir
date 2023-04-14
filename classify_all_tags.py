import numpy as np
import argparse
import pathlib
import random
import os

from sklearn.ensemble import RandomForestClassifier
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
    random.seed(0)
    random.shuffle(npy_paths)

    # Loads data
    X = np.array([np.load(p) for p in npy_paths])
    labels = np.array([os.path.split(p)[0].split('/')[-1] for p in npy_paths])
    # one example label: electronic-neither-expressive-neither-rugged-human-neither-neither-neither-neither

    # One-hot encoding
    label_map = {'acoustic': [1, 0, 0], 'electronic': [0, 1, 0], 'neither': [0, 0, 1],
                 'careful': [1, 0, 0], 'provocative': [0, 1, 0],
                 'discrete': [1, 0, 0], 'expressive': [0, 1, 0],
                 'downtoearth': [1, 0, 0], 'dreamy': [0, 1, 0],
                 'elegant': [1, 0, 0], 'rugged': [0, 1, 0],
                 'human': [1, 0, 0], 'technological': [0, 1, 0],
                 'inclusive': [1, 0, 0], 'exclusive': [0, 1, 0],
                 'modern': [1, 0, 0], 'traditional': [0, 1, 0],
                 'serious': [1, 0, 0], 'easygoing': [0, 1, 0],
                 'youthful': [1, 0, 0], 'mature': [0, 1, 0]}
    y = np.zeros((len(labels), 30)) # evaluating all 10 dimensions
    for i, label in enumerate(labels):
        label_parts = label.split('-')
        for j, part in enumerate(label_parts):
            y[i, j*3:j*3+3] = label_map[part]

    # Train test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    print("X train and test shapes:", X_train.shape, X_test.shape)
    print("y train and test shapes:", y_train.shape, y_test.shape)

    # Random forest classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # # Predictions
    # y_pred = clf.predict(X_test)
    # from sklearn.metrics import f1_score
    # print(f1_score(y_test, y_pred, average='weighted', zero_division=1))
    # Predictions
    y_pred = clf.predict(X_test)

    # Evaluate predictions with ROC
    from sklearn.metrics import roc_auc_score
    roc_score = roc_auc_score(y_test, y_pred)
    print("ROC score:", roc_score)
