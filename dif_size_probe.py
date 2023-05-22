import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sklearn.metrics as metrics
import numpy as np
import argparse
import pathlib
import os
import random
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import json

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

def load_data(input_dir, dimension):
    npy_paths = sorted(list(input_dir.glob("**/*.npy")))

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
    random.seed(3)
    indices = np.arange(len(new_X))
    random.shuffle(indices)
    new_X = new_X[indices]
    new_y = new_y[indices]

    # Normalize the data
    X_mean = np.mean(new_X, axis=0)
    X_std = np.std(new_X, axis=0)
    new_X = (new_X - X_mean) / X_std

    # Convert labels to 0 and 1
    new_y = np.array([0 if label == unique[0] else 1 for label in new_y])

    return new_X, new_y


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
    X, y = load_data(input_dir, args.dimension)
    input_dim = X.shape[1]

    # Set up data sizes to be used
    data_sizes = [20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
    unique, counts = np.unique(y, return_counts=True)

    # Only include data sizes smaller than the max number of samples per class (200 samples are used for testing)
    upper_limit = min(counts) - 200
    data_sizes = [x for x in data_sizes if x < upper_limit]

    # Add the max number of samples per class
    data_sizes.append(int(upper_limit))

    f1_scores = [0]*len(data_sizes)
    i = 0
    model_file = f"best_model.pth"

    #if torch.backends.mps.is_available():
        # device = torch.device("mps")
    #elif torch.cuda.is_available():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # print("Number of samples per class:")
    # for label, count in zip(unique, counts):
    #     print(f"{label}: {count}")

    # take 100 samples from each class and make that a test set, that won't change
    test_X = []
    test_y = []
    for label in unique:
        indices = np.where(y == label)[0]
        test_X.append(X[indices[:100]])
        test_y.append(y[indices[:100]])

        # remove these from X and y
        X = np.delete(X, indices[:100], axis=0)
        y = np.delete(y, indices[:100], axis=0)

    test_X = np.concatenate(test_X)
    test_y = np.concatenate(test_y)

    test_X_tensor = torch.from_numpy(test_X).float().to(device)


    for n_samples in data_sizes:
        random.seed(3)
        new_X = []
        new_y = []
        for label in unique:
            indices = np.where(y == label)[0]
            random.shuffle(indices)
            new_X.append(X[indices[:n_samples]])
            new_y.append(y[indices[:n_samples]])

        new_X = np.concatenate(new_X)
        new_y = np.concatenate(new_y)

        # Set up for k-fold cross-validation
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits)


        # Implement sk-fold cross-validation
        for fold, (train_index, val_index) in enumerate(skf.split(new_X, new_y)):

            X_train, X_val = new_X[train_index], new_X[val_index]
            y_train, y_val = new_y[train_index], new_y[val_index]

            # Convert numpy arrays to PyTorch tensors
            X_train_tensor = torch.from_numpy(X_train).float().to(device)
            y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1).to(device)
            X_val_tensor = torch.from_numpy(X_val).float().to(device)
            y_val_tensor = torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1).to(device)

            # Define datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            # Define data loaders
            min_batch_size = 256
            batch_size = min(min_batch_size, len(train_dataset))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # train model
            net = Net(input_dim).to(device)

            criterion = nn.BCEWithLogitsLoss()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

            patience = 50
            best_val_loss = float("inf")
            wait = 0

            for epoch in range(int(args.epochs)):
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch, y_batch
                    optimizer.zero_grad()
                    output = net(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()

                if epoch % 10 == 0:

                    #val_output = net(torch.from_numpy(X_val).float().to(device))
                    val_output = net(X_val_tensor)
                    #val_error = criterion(val_output, torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1).to(device))
                    val_error = criterion(val_output, y_val_tensor)
                    current_val_loss = val_error.item()

                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        net = net.to("cpu")
                        torch.save(net.state_dict(), model_file)
                        net = net.to(device)
                        wait = 0

                    else:
                        wait += 1
                        if args.early_stopping and wait >= patience:
                            break

            net.load_state_dict(torch.load(model_file))
            net.eval()
            #y_pred = net(torch.from_numpy(test_X).float().to(device))
            y_pred = net(test_X_tensor)
            y_pred = torch.round(torch.sigmoid(y_pred))
            f1_score = metrics.f1_score(test_y, y_pred.cpu().detach().numpy())
            f1_scores[i] += f1_score

        f1_scores[i] = f1_scores[i]/n_splits
        # print f1 score for data size of n_samples
        print(f'Number of samples: {n_samples}')
        print(f'f1 score: {f1_scores[i]}')

        i += 1
    input_str = str(input_dir)
    name_of_model = input_str.split('/')[1].replace('_features', '')

    data_to_save = {'data_sizes': data_sizes, 'f1_scores': f1_scores, 'model': name_of_model}
    # save data, check if there is a run_0 file already, if so, save as run_1, etc.
    i = 0
    while os.path.exists(f"data/datasizes/run_{i}.json"):
        i += 1
    with open(f"data/datasizes/run_{i}.json", 'w') as f:
        json.dump(data_to_save, f)

    # sns.lineplot(x=data_sizes, y=f1_scores, label="f1 score")
    # plt.legend()
    # plt.show()

# # openai
# # python dif_size_probe.py data/openai_features/music_tags 0 --learning_rate 0.00016376921949021333 --weight_decay 0.0013168925817661605 --early_stopping --dropout_prob 0.48355187571656977
#
# # choi
# # python dif_size_probe.py data/choi_features/music_tags 0 --learning_rate 0.004995732273615719 --weight_decay 0.001268951728049984 --early_stopping --dropout_prob 0.4059396056893086
#
# # l3net
# # python dif_size_probe.py data/l3net_features/music_tags 0 --learning_rate 0.0009615673995981056 --weight_decay 0.051582663771638546 --early_stopping --dropout_prob 0.3389827761755996
#
# # jukebox
# # python dif_size_probe.py data/jukebox_features/music_tags 0 --learning_rate 0.0080705879972147 --weight_decay 0.014420622945033037 --early_stopping --dropout_prob 0.2848565484101526
#
# # combined
# # python dif__size_probe.py data/combined_features/music_tags 0 --learning_rate 0.006565155696330251 --weight_decay 0.04934280167439439 --early_stopping --dropout_prob 0.32808548973867613
