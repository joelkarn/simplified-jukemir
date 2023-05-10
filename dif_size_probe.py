import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
import numpy as np
import argparse
import pathlib
import os
import random
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(p=args.dropout_prob)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x

def train_model(config):
    args.learning_rate = config["learning_rate"]
    args.dropout_prob = config["dropout_prob"]
    args.weight_decay = config["weight_decay"]

    input_dim = X.shape[1]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files.")

    parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
    parser.add_argument("dimension", type=str, help="Which dimension (0-9) to investigate.")

    parser.add_argument("--epochs", type=float, default=10000, help="Number of epochs to train for.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--dropout_prob", type=float, default=0, help="Dropout probability.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer.")
    parser.add_argument("--early_stopping", action='store_true', help="Enable early stopping.")

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)

    npy_paths = sorted(list(input_dir.glob("**/*.npy")))

    # Loads data
    X = np.array([np.load(p) for p in npy_paths])
    y = np.array([os.path.split(p)[0].split('/')[-1].split('-')[int(args.dimension)] for p in npy_paths])

    input_dim = X.shape[1]

    # remove all samples with label 'neither'
    indices = np.where(y == 'neither')[0]
    X = np.delete(X, indices, axis=0)
    y = np.delete(y, indices, axis=0)

    # Normalize the data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    # Normalize the data
    X = (X - X_mean) / X_std

    # make new X and y with equal number of samples per class
    unique, counts = np.unique(y, return_counts=True)
    data_sizes = list(range(100, counts.min(), 200))
    f1_scores = [0]*len(data_sizes)
    i = 0
    model_file = f"best_model.pth"

    for n_samples in data_sizes:
        random.seed(i)
        new_X = []
        new_y = []
        for label in unique:
            indices = np.where(y == label)[0]
            random.shuffle(indices)
            new_X.append(X[indices[:n_samples]])
            new_y.append(y[indices[:n_samples]])

        new_X = np.concatenate(new_X)
        new_y = np.concatenate(new_y)

        # there are two classes, so make labels 0 and 1
        new_y = np.array([0 if label == unique[0] else 1 for label in new_y])

        # Set up for k-fold cross-validation
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits)

        # For storing average performance across all folds
        best_val_losses = []

        # Implement k-fold cross-validation
        for fold, (train_index, val_index) in enumerate(skf.split(new_X, new_y)):

            X_train, X_val = new_X[train_index], new_X[val_index]
            y_train, y_val = new_y[train_index], new_y[val_index]

            # train model
            net = Net(input_dim)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

            patience = 50
            best_val_loss = float("inf")
            wait = 0

            for epoch in range(int(args.epochs)):
                optimizer.zero_grad()
                output = net(torch.from_numpy(X_train).float())
                loss = criterion(output, torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1))
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:

                    val_output = net(torch.from_numpy(X_val).float())
                    val_error = criterion(val_output, torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1))
                    current_val_loss = val_error.item()

                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        torch.save(net.state_dict(), model_file)
                        wait = 0

                    else:
                        wait += 1
                        if args.early_stopping and wait >= patience:
                            break

            net.load_state_dict(torch.load(model_file))
            net.eval()
            y_pred = net(torch.from_numpy(X_val).float())
            y_pred = torch.round(torch.sigmoid(y_pred))
            f1_score = metrics.f1_score(y_val, y_pred.detach().numpy())
            f1_scores[i] += f1_score

        f1_scores[i] = f1_scores[i]/n_splits
        # print f1 score for data size of n_samples
        print(f'Number of samples: {n_samples}')
        print(f'f1 score: {f1_scores[i]}')

        i += 1

    print(f1_scores)
    sns.lineplot(x=data_sizes, y=f1_scores, label="f1 score")
    plt.legend()
    plt.show()

# choi
# python dif_size_probe.py data/choi_features/music_tags 0 --learning_rate 0.004995732273615719 --weight_decay 0.001268951728049984 --early_stopping --dropout_prob 0.4059396056893086