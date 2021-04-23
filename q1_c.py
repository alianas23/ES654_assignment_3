import pandas as pd
import numpy as np
from log_reg import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer

data = np.array(load_breast_cancer().data)
y = np.array(load_breast_cancer().target)
k = KFold(n_splits=3)
for train_idx,test_idx in k.split(data):
    X_train,X_test = data[train_idx], data[test_idx]
    y_train,y_test = y[train_idx], y[test_idx]

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)
clf1 = LogisticRegression().fit(X_train, y_train)
y_hat = list(clf1.predict(X_test))
y_GT = list(y_test)
count = 0
for i in range(len(y_hat)):
    if int(y_hat[i]) == (y_GT[i]):
        count+=1
print("accuracy for unregularised Logisitc Regression:{}".format(100*count/len(y_hat)))