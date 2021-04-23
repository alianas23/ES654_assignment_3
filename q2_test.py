from q2_LR import LogisticRegression
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
X, y = make_classification(n_samples=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = LogisticRegression().l1_fit(X_train, y_train)
y_hat = (clf.predict(X_test))
count = 0
for i in range((len(y_hat))):
    if int(y_hat[i]) == y[i]:
        count += 1
accuracy =  count/len(y_hat)       
print("Accuracy (L1 regularised)")
print(accuracy)

clf = LogisticRegression().l2_fit(X_train, y_train)
y_hat = (clf.predict(X_test))
count = 0
for i in range((len(y_hat))):
    if int(y_hat[i]) == y[i]:
        count += 1
accuracy =  count/len(y_hat)       
print("Accuracy (L2 regularised)")
print(accuracy)

