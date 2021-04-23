from q2_LR import LogisticRegression
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=100, random_state=np.random.RandomState(42))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = LogisticRegression(l1_coef = 5).l1_fit(X_train, y_train)
theta = clf.get_theta()


x=[]
y = []
for i in range(len(theta)):
    x.append('θ'+str(i))
    y.append(abs(theta[i][0]))
        
plt.bar(x,y)
plt.title("important features")
plt.xlabel("thetas")
plt.ylabel("magnitude of thetas")
plt.show()