import numpy as np
import argparse
import pathlib
import random
import os
import seaborn as sns
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
    print("HEJSAN!!!!!")
    npy_paths = sorted(list(input_dir.glob("**/*.npy")))
    random.seed(0)
    random.shuffle(npy_paths)

    # Loads data
    X = np.array([np.load(p) for p in npy_paths])
    y = np.array([os.path.split(p)[0].split('/')[-1] for p in npy_paths])

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
    #clf = make_pipeline(StandardScaler(), RandomForestClassifier())
    scores = cross_val_score(clf, new_X, new_y, cv=10, scoring='f1_macro')
    y_pred = cross_val_predict(clf, new_X, new_y, cv=10)
    conf_mat = confusion_matrix(new_y, y_pred)

    #print('{:.1f} +- {:.1f}'.format(np.mean(scores) * 100, np.std(scores) * 100))
    print(conf_mat)

    # Plot confusion matrix with labels
    labels = sorted(set(y))
    # from sklearn.metrics import classification_report
    # report = classification_report(new_y, y_pred, target_names=labels)
    # print(report)
    # with open('data/classification_reports/genres/classification_report.txt', 'w') as file:
    #     file.write(report)

    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    # disp.plot()
    # Use Seaborn's heatmap function to plot the confusion matrix
    plt.figure(figsize=(10, 7))  # Adjust the figure size as needed
    sns.set(font_scale=1.2)  # Adjust the font scale as needed

    if "bert" in input_dir.name:
        color = "Greys"
    if "openai" in input_dir.name:
        color = "Reds"
    if "choi" in input_dir.name:
        color = "Oranges"
    if "l3net" in input_dir.name:
        color = "Greens"
    if "juke" in input_dir.name:
        color = "Blues"
    if "openai_combined" in input_dir.name:
        color = "Purples"


    sns.heatmap(conf_mat, annot=True, fmt="d", cmap=color, xticklabels=labels, yticklabels=labels)

    # Set labels for x and y axis
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Save plot as PDF file
    plot_name = 'data/conf_mat/genres/{:.2f}_{:.2f}_cv10.pdf'.format(np.mean(scores) * 100, np.std(scores) * 100)
    plt.savefig(plot_name)
