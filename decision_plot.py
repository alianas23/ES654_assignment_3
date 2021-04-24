from log_reg import LogisticRegression
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as cma

data = pd.DataFrame(load_breast_cancer().data)
df = data.sample(n=2,axis='columns')
y = pd.Series(load_breast_cancer().target)

def plot_decision_boundary(X, y, model):
    cMap = cma.ListedColormap(["#6b76e8", "#c775d1"])
    cMapa = cma.ListedColormap(["#6b76e8", "#c775d1"])

    x_min, x_max = X.iloc[:, 0].min() - .5, X.iloc[:, 0].max() + .5
    y_min, y_max = X.iloc[:, 1].min() - .5, X.iloc[:, 1].max() + .5
    h = .02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
    Z = Z.reshape(xx.shape)

    plt.figure(1, figsize=(8, 6), frameon=True)
    plt.axis('off')
    plt.pcolormesh(xx, yy, Z, cmap=cMap)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, marker = "o", edgecolors='k', cmap=cMapa)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

clf = LogisticRegression().fit(df, y)
plot_decision_boundary(df, y, clf)