import ray
from ray import tune
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import argparse
import pathlib
import os
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files.")

    parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
    parser.add_argument("dimension", type=str, help="Which dimension (0-9) to investigate.")

    parser.add_argument("--epochs", type=float, default=1000, help="Number of epochs to train for.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--step_size", type=float, default=100, help="Step size for the scheduler.")
    parser.add_argument("--gamma", type=float, default=1, help="Gamma for the scheduler.")
    parser.add_argument("--dropout_prob", type=float, default=0, help="Dropout probability.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for the optimizer.")
    parser.add_argument("--early_stopping", action='store_true', help="Enable early stopping.")

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)

    npy_paths = sorted(list(input_dir.glob("**/*.npy")))
    random.seed(0)
    random.shuffle(npy_paths)

    # Loads data
    X = np.array([np.load(p) for p in npy_paths])
    y = np.array([os.path.split(p)[0].split('/')[-1].split('-')[int(args.dimension)] for p in npy_paths])

    input_dim = X.shape[1]

    # remove all samples with label 'neither'
    indices = np.where(y == 'neither')[0]
    X = np.delete(X, indices, axis=0)
    y = np.delete(y, indices, axis=0)

    # make new X and y with equal number of samples per class
    unique, counts = np.unique(y, return_counts=True)
    n_samples = min(counts)

    new_X = []
    new_y = []
    for label in unique:
        indices = np.where(y == label)[0]
        new_X.append(X[indices[:n_samples]])
        new_y.append(y[indices[:n_samples]])

    new_X = np.concatenate(new_X)
    new_y = np.concatenate(new_y)

    # shuffle data
    indices = np.arange(len(new_X))
    np.random.shuffle(indices)
    new_X = new_X[indices]
    new_y = new_y[indices]
    # there are two classes, so make labels 0 and 1
    new_y = np.array([0 if label == unique[0] else 1 for label in new_y])

    # Normalize the data
    X_mean = np.mean(new_X, axis=0)
    X_std = np.std(new_X, axis=0)

    # Normalize the data
    X_norm = (new_X - X_mean) / X_std

    # Add these lines before the training loop
    n_splits = 5
    kf = KFold(n_splits=n_splits)

    # For storing average performance across all folds
    all_train_errors = []
    all_val_errors = []
    best_val_losses = []

    # Implement k-fold cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(X_norm)):
        print(f"Fold {fold + 1}")

        X_train, X_val = X_norm[train_index], X_norm[val_index]
        y_train, y_val = new_y[train_index], new_y[val_index]

        # train model
        net = Net(input_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        patience = 100
        best_val_loss = float("inf")
        wait = 0
        model_file = f"best_model_fold_{fold + 1}.pth"

        train_errors = []
        val_errors = []
        for epoch in range(int(args.epochs)):
            optimizer.zero_grad()
            output = net(torch.from_numpy(X_train).float())
            loss = criterion(output, torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 10 == 0:
                print("Epoch: {}".format(epoch))
                train_errors.append(loss.item())
                print(f'Training error: {loss.item()}')

                val_output = net(torch.from_numpy(X_val).float())
                val_error = criterion(val_output, torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1))
                val_errors.append(val_error.item())
                print(f'Validation error: {val_error.item()}')

                current_val_loss = val_error.item()
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    torch.save(net.state_dict(), model_file)
                    wait = 0
                else:
                    wait += 1
                    if args.early_stopping and wait >= patience:
                        print("Early stopping")
                        break
        best_val_losses.append(best_val_loss)
        all_train_errors.append(train_errors)
        all_val_errors.append(val_errors)

        # Load the best model for evaluation or further use
        net.load_state_dict(torch.load(model_file))

    # Find the maximum number of epochs across all folds
    max_epochs = max(max(len(train_errs) for train_errs in all_train_errors), max(len(val_errs) for val_errs in all_val_errors))

    # Pad the lists of errors with None values if the training stopped early
    for train_errs, val_errs in zip(all_train_errors, all_val_errors):
        if len(train_errs) < max_epochs:
            train_errs += [None] * (max_epochs - len(train_errs))
        if len(val_errs) < max_epochs:
            val_errs += [None] * (max_epochs - len(val_errs))

    # Calculate the average errors for each epoch while ignoring None values
    avg_train_errors = [np.mean([err for err in train_errs if err is not None]) for train_errs in zip(*all_train_errors)]
    avg_val_errors = [np.mean([err for err in val_errs if err is not None]) for val_errs in zip(*all_val_errors)]
    print(best_val_losses)
    plt.plot(avg_train_errors, label='average training error')
    plt.plot(avg_val_errors, label='average validation error')
    plt.legend()
    plt.show()

    # # choi - lr=0.01, weight_decay=7e-3, step_size=100, gamma=0.9, no dropout
    # python probe.py data/choi_features/music_tags 0 --epochs 10000 --learning_rate 0.1 --step_size 100 --gamma 0.9 --weight_decay 0.007 --early_stopping --dropout_prob 0.3

    # # l3 - lr=0.001, weight_decay=5e-2, step_size=500, gamma=0.95, dropout=0.4
    #python probe.py data/l3net_features/music_tags 0 --epochs 10000 --learning_rate 0.0001 --step_size 100 --gamma 1 --weight_decay 0.2 --early_stopping --dropout_prob 0.45

    # # juke - lr=0.001, weight_decay=5e-2, step_size=500, gamma=0.95, dropout=0.4
    # # combined - 0.001