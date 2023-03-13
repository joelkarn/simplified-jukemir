import glob
import os
import random

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Find numpy paths (and randomize to remove label ordering)
npy_paths = sorted(glob.glob('features/*.npy'))
random.seed(0)
random.shuffle(npy_paths)

# Loads data
X = np.array([np.load(p) for p in npy_paths])
y = np.array([os.path.split(p)[1].split('.')[0] for p in npy_paths])

print(X.shape, y)

# # Run cross-validation
# clf = make_pipeline(StandardScaler(), SVC())
# scores = cross_val_score(clf, X, y, cv=10)
# print('{:.1f} +- {:.1f}'.format(np.mean(scores) * 100, np.std(scores) * 100))