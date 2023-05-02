# import numpy as np
# import argparse
# import pathlib
# import random
# import os
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Classify all 10 music tags per song.")
#     parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
#     args = parser.parse_args()
#     input_dir = pathlib.Path(args.input_dir)
#
#     npy_paths = sorted(list(input_dir.glob("**/*.npy")))
#     random.seed(0)
#     random.shuffle(npy_paths)
#
#     # Loads data
#     X = np.array([np.load(p) for p in npy_paths])
#
#     # Normalize data
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     X = (X - mean) / std
#
#     labels = np.array([os.path.split(p)[0].split('/')[-1] for p in npy_paths])
#     # one example label: electronic-neither-expressive-neither-rugged-human-neither-neither-neither-neither
#
#     # One-hot encoding
#     label_map = {'acoustic': [1, 0, 0], 'electronic': [0, 1, 0], 'neither': [0, 0, 1],
#                  'careful': [1, 0, 0], 'provocative': [0, 1, 0],
#                  'discrete': [1, 0, 0], 'expressive': [0, 1, 0],
#                  'downtoearth': [1, 0, 0], 'dreamy': [0, 1, 0],
#                  'elegant': [1, 0, 0], 'rugged': [0, 1, 0],
#                  'human': [1, 0, 0], 'technological': [0, 1, 0],
#                  'inclusive': [1, 0, 0], 'exclusive': [0, 1, 0],
#                  'modern': [1, 0, 0], 'traditional': [0, 1, 0],
#                  'serious': [1, 0, 0], 'easygoing': [0, 1, 0],
#                  'youthful': [1, 0, 0], 'mature': [0, 1, 0]}
#     y = np.zeros((len(labels), 30)) # evaluating all 10 dimensions
#     for i, label in enumerate(labels):
#         label_parts = label.split('-')
#         for j, part in enumerate(label_parts):
#             y[i, j*3:j*3+3] = label_map[part]
#     print(labels[0]) # example: neither-careful-discrete-neither-neither-human-exclusive-traditional-neither-mature
#     print(y[0]) # example: [0. 0. 1. 1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 1. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1. 0. 1. 0.]
#     # # Train test split
#     # from sklearn.model_selection import train_test_split
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#     #
#     # print("X train and test shapes:", X_train.shape, X_test.shape)
#     # print("y train and test shapes:", y_train.shape, y_test.shape)
#     #
#     # # Random forest classifier
#     # from sklearn.ensemble import RandomForestClassifier
#     # clf = RandomForestClassifier(n_estimators=100)
#     # clf.fit(X_train, y_train)
#     #
#     # # # Predictions
#     # # y_pred = clf.predict(X_test)
#     # # from sklearn.metrics import f1_score
#     # # print(f1_score(y_test, y_pred, average='weighted', zero_division=1))
#     # # Predictions
#     # y_pred = clf.predict(X_test)
#     #
#     # # Evaluate predictions with ROC
#     # from sklearn.metrics import roc_auc_score
#     # roc_score = roc_auc_score(y_test, y_pred)
#     # print("ROC score:", roc_score)


# import argparse
# import pathlib
# import random
# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow import keras
#
# # Define model architecture
# model = keras.Sequential([
#     keras.layers.Dense(64, activation='relu', input_shape=(10,)),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(30, activation='softmax')
# ])
#
# # Define loss function and optimizer
# loss_fn = keras.losses.CategoricalCrossentropy()
# optimizer = keras.optimizers.Adam()
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Classify all 10 music tags per song.")
#     parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
#     args = parser.parse_args()
#     input_dir = pathlib.Path(args.input_dir)
#
#     npy_paths = sorted(list(input_dir.glob("**/*.npy")))
#     random.seed(0)
#     random.shuffle(npy_paths)
#
#     # Loads data
#     X = np.array([np.load(p) for p in npy_paths])
#
#     # Normalize data
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     X = (X - mean) / std
#
#     labels = np.array([os.path.split(p)[0].split('/')[-1] for p in npy_paths])
#     # one example label: electronic-neither-expressive-neither-rugged-human-neither-neither-neither-neither
#
#     # One-hot encoding
#     label_map = {'acoustic': [1, 0, 0], 'electronic': [0, 1, 0], 'neither': [0, 0, 1],
#                  'careful': [1, 0, 0], 'provocative': [0, 1, 0],
#                  'discrete': [1, 0, 0], 'expressive': [0, 1, 0],
#                  'downtoearth': [1, 0, 0], 'dreamy': [0, 1, 0],
#                  'elegant': [1, 0, 0], 'rugged': [0, 1, 0],
#                  'human': [1, 0, 0], 'technological': [0, 1, 0],
#                  'inclusive': [1, 0, 0], 'exclusive': [0, 1, 0],
#                  'modern': [1, 0, 0], 'traditional': [0, 1, 0],
#                  'serious': [1, 0, 0], 'easygoing': [0, 1, 0],
#                  'youthful': [1, 0, 0], 'mature': [0, 1, 0]}
#     y = np.zeros((len(labels), 30)) # evaluating all 10 dimensions
#     for i, label in enumerate(labels):
#         label_parts = label.split('-')
#         for j, part in enumerate(label_parts):
#             y[i, j*3:j*3+3] = label_map[part]
#
#     # Split data into training and validation sets
#     train_size = int(0.8 * len(X))
#     X_train, y_train = X[:train_size], y[:train_size]
#     X_val, y_val = X[train_size:], y[train_size:]
#
#     # Compile the model
#     model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
#
#     # Train the model
#     history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
# import argparse
# import os
# import pathlib
# import random
#
# import numpy as np
# import tensorflow as tf
#
# from sklearn.model_selection import train_test_split
#
# def build_model(input_shape, output_shape):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Dense(512, input_shape=input_shape, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.3))
#     model.add(tf.keras.layers.Dense(256, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.3))
#     model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid'))
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Classify all 10 music tags per song.")
#     parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
#     args = parser.parse_args()
#     input_dir = pathlib.Path(args.input_dir)
#
#     npy_paths = sorted(list(input_dir.glob("**/*.npy")))
#     random.seed(0)
#     random.shuffle(npy_paths)
#
#     # Loads data
#     X = np.array([np.load(p) for p in npy_paths])
#
#     # Normalize data
#     mean = np.mean(X, axis=0)
#     std = np.std(X, axis=0)
#     X = (X - mean) / std
#
#     labels = np.array([os.path.split(p)[0].split('/')[-1] for p in npy_paths])
#
#     # One-hot encoding
#     label_map = {'acoustic': [1, 0, 0], 'electronic': [0, 1, 0], 'neither': [0, 0, 1],
#                  'careful': [1, 0, 0], 'provocative': [0, 1, 0],
#                  'discrete': [1, 0, 0], 'expressive': [0, 1, 0],
#                  'downtoearth': [1, 0, 0], 'dreamy': [0, 1, 0],
#                  'elegant': [1, 0, 0], 'rugged': [0, 1, 0],
#                  'human': [1, 0, 0], 'technological': [0, 1, 0],
#                  'inclusive': [1, 0, 0], 'exclusive': [0, 1, 0],
#                  'modern': [1, 0, 0], 'traditional': [0, 1, 0],
#                  'serious': [1, 0, 0], 'easygoing': [0, 1, 0],
#                  'youthful': [1, 0, 0], 'mature': [0, 1, 0]}
#     y = np.zeros((len(labels), 30)) # evaluating all 10 dimensions
#     for i, label in enumerate(labels):
#         label_parts = label.split('-')
#         for j, part in enumerate(label_parts):
#             y[i, j*3:j*3+3] = label_map[part]
#
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
#     # Build and fit the model
#     model = build_model(input_shape=(X_train.shape[1],), output_shape=y_train.shape[1])
#     model.summary()
#     history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
import argparse
import os
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 512)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, output_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify all 10 music tags per song.")
    parser.add_argument("input_dir", type=str, help="Path to feature parent directory.")
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)

    npy_paths = sorted(list(input_dir.glob("**/*.npy")))
    random.seed(0)
    random.shuffle(npy_paths)

    # Loads data
    X = np.array([np.load(p) for p in npy_paths])

    # Normalize data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    labels = np.array([os.path.split(p)[0].split('/')[-1] for p in npy_paths])
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

    y = torch.tensor(y, dtype=torch.float32)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Build and fit the model
    model = Net(input_shape=X_train.shape[1], output_shape=y_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))