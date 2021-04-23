import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics

df = datasets.load_digits()
X = df.data
y = df.target
k = model_selection.StratifiedKFold(n_splits=4)
for train_idx, test_idx in k.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 0)

clf = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
clf.fit(X_train, y_train)
print("Accuracy = %s" %(clf.score(X_test, y_test)))

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Confusion Matrx')

disp =metrics.plot_confusion_matrix(clf, X_test, y_test, display_labels= df.target_names, ax = ax)
disp.confusion_matrix
plt.show()