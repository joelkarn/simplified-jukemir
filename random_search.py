import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pathlib
import os
import random
from sklearn.model_selection import StratifiedKFold
import ray
from ray import tune


class Net(nn.Module):
    def __init__(self, input_dim, dropout_prob):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x


def load_data(input_dir, dimension):
    npy_paths = sorted(list(input_dir.glob("**/*.npy")))
    random.seed(0)
    random.shuffle(npy_paths)

    # Loads data
    X = np.array([np.load(p) for p in npy_paths])
    y = np.array([os.path.split(p)[0].split('/')[-1].split('-')[int(dimension)] for p in npy_paths])

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

    return X_norm, new_y


def train_model(config, X, y):
    learning_rate = config["learning_rate"]
    dropout_prob = config["dropout_prob"]
    weight_decay = config["weight_decay"]

    input_dim = X.shape[1]

    epochs = 10000
    # Add these lines before the training loop
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits)

    best_val_losses = []

    # Implement k-fold cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # train model
        net = Net(input_dim, dropout_prob)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        patience = 50
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(epochs):
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
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
        best_val_losses.append(best_val_loss)

    average_val_loss = np.mean(best_val_losses)
    # Report the final validation error to Ray Tune
    tune.report(final_val_error=average_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files.")

    parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
    parser.add_argument("dimension", type=str, help="Which dimension (0-9) to investigate.")
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)
    dimension = args.dimension

    X, y = load_data(input_dir, dimension)

    # Define the search space for the hyperparameters
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "dropout_prob": tune.uniform(0, 0.5),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
    }

    ray.init(include_dashboard=False)
    analysis = tune.run(
        lambda cfg: train_model(cfg, X, y),
        config=search_space,
        num_samples=20,  # Number of hyperparameter configurations to try
        resources_per_trial={"cpu": 2},  # CPU resources to allocate per trial
        local_dir="data",  # Directory to store Ray Tune results
    )

    # Get the best configuration and print it
    best_trial = analysis.get_best_trial("final_val_error", "min", "last")
    print("Best trial configuration: {}".format(best_trial.config))