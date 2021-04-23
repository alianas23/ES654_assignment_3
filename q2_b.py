from q2_LR import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300)
X= pd.DataFrame(X)
y = pd.Series(y)
accuracy=[]
x = []
for j in range(50):
    a = []
    for i in range(6):
        X1 = X[0:i*50]
        X2 = X[(i+1)*50:]
        X_train = X1.append(X2)
        X_test = X[i*50:(i+1)*50]
        y1 = y[0:i*50]
        y2 = y[(i+1)*50:]
        y_train = y1.append(y2)
        y_test = y[i*50:(i+1)*50]
        clf = LogisticRegression(l2_coef= j*2).l2_fit(X_train,y_train)
        y_hat = clf.predict(X_test)
        y_GT = list(y_test)
        count = 0
        for i in range(len(y_GT)):
            if int(y_hat[i]) == y_GT[i]:
                count+=1
        a.append((count)/len(y_GT))
    accuracy.append((sum(a)/len(a)))
    x.append(j*2)
    # print(a)
print(accuracy)
plt.xlabel("Penalty")
plt.ylabel("Accuracy")
plt.plot(x,accuracy)
plt.show()


