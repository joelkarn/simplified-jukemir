import numpy as np
import argparse
import pathlib
import random
import os
import json
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

random.seed(1337)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files.")

    parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
    parser.add_argument("dimension", type=str, help="Which dimension (0-9) to investigate.")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)

    npy_paths = sorted(list(input_dir.glob("**/*.npy")))
    dim = args.dimension

    # Loads and shuffles data
    X = np.array([np.load(p) for p in npy_paths])
    y = np.array([os.path.split(p)[0].split('/')[-1].split('-')[int(dim)] for p in npy_paths])
    # remove all samples with label 'neither'
    indices = np.where(y == 'neither')[0]
    X = np.delete(X, indices, axis=0)
    y = np.delete(y, indices, axis=0)

    unique, counts = np.unique(y, return_counts=True)
    upper_limit = min(counts) - 200

    test_X = []
    test_y = []

    # select 100 random samples from each class and put in a test set
    for label in unique:
        indices = np.where(y == label)[0]
        random.shuffle(indices)
        test_X.append(X[indices[:100]])
        test_y.append(y[indices[:100]])

        # remove the 100 samples from X and y
        X = np.delete(X, indices[:100], axis=0)
        y = np.delete(y, indices[:100], axis=0)

    test_X = np.concatenate(test_X)
    test_y = np.concatenate(test_y)
    data_sizes = list(range(1, 10, 1))
    data_sizes.extend(list(range(10, 100, 10)))
    data_sizes.extend(list(range(100, upper_limit, 100)))
    data_sizes.append(int(upper_limit))
    output_data = []

    i = 0
    for n_samples in data_sizes:
        for j in range(90):
            # make new X and y with equal number of samples per class
            new_X = []
            new_y = []
            for label in unique:
                indices = np.where(y == label)[0]
                random.shuffle(indices)
                new_X.append(X[indices[:n_samples]])
                new_y.append(y[indices[:n_samples]])

            new_X = np.concatenate(new_X)
            new_y = np.concatenate(new_y)

            clf = make_pipeline(StandardScaler(), SVC(probability=True, random_state=j))
            #clf = make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000))
            clf.fit(new_X, new_y)

            #test_y_pred = clf.predict(test_X)
            test_y_prob = clf.predict_proba(test_X)[:, 1]
            auc_roc_score = roc_auc_score(test_y, test_y_prob)
            #f1_score = metrics.f1_score(test_y, test_y_pred, average='macro')
            data_point = {'n_samples': n_samples, 'auc_roc_score': auc_roc_score}
            output_data.append(data_point)
        print(f"n_samples: {n_samples}, auc_roc_score: {auc_roc_score}")
        i += 1

    # save data, check if there is a run_0 file already, if so, save as run_1, etc.
    input_str = str(input_dir)
    name_of_model = input_str.split('/')[1].replace('_features', '')
    data_to_save = {'model': name_of_model, 'dimension': dim, 'data': output_data}
    i = 0
    while os.path.exists(f"data/svm_datasizes/auc/inclusive-exclusive/run_{i}.json"):
        i += 1
    with open(f"data/svm_datasizes/auc/inclusive-exclusive/run_{i}.json", 'w') as f:
        json.dump(data_to_save, f)
