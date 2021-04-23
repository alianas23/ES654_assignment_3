from log_reg import LogisticRegression
import pandas as pd
import numpy as np

N = 50
P = 8
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(0,2,N))

clf = LogisticRegression().fit(X, y)
y_hat = (clf.predict(X))

count = 0
for i in range(len(y)):
    if int(y_hat[i]) == (y[i]):
        count+=1
acc = (count/len(y_hat))
print("Accuracy Normally")
print(acc)

clf = LogisticRegression().fit_autograd(X, y)
y_hat = (clf.predict(X))
count = 0
for i in range(len(y)):
    if int(y_hat[i]) == (y[i]):
        count+=1
acc = (count/len(y_hat))
print("Accuracy uisng AutoGrad")
print(acc)
