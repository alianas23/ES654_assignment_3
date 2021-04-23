import autograd.numpy as np
from autograd import grad
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy
sns.set(style="white")

class LogisticRegression(object):

    def __init__(self, learning_rate=0.01, max_iter=50, l1_coef = 0.5, l2_coef = 0.5):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1] + 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        for _ in range(self.max_iter):
            y_hat = self.sigmoid(np.dot(X,self.theta))
            errors = y_hat - y
            N = X.shape[1]

            delta_grad = self.learning_rate * (np.dot(X.T,errors))
            self.theta -= delta_grad / N
                
        return self
    
    def fit_autograd(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        X=np.array(X).astype(np.float64)
        y=np.array(y).reshape(-1,1).astype(np.float64)
        self.theta=np.zeros(X.shape[1]).reshape(-1,1)
        def loss(theta,X,y):
            y_hat = self.sigmoid(np.dot(X,theta))
            y_hat = np.squeeze(y_hat)
            y = np.squeeze(y)
            res = -np.sum(y.dot(np.log10(y_hat))+(1-y).dot(np.log10(1-y_hat)))
            res = res/X.shape[0]
            return res
        delta = grad(loss)

        for _ in range(self.max_iter):
            gradient = delta(self.theta,X,y)
            self.theta -= self.learning_rate*gradient
            

        return self

    def predict_proba(self, X):
        prob =  self.theta[0] + np.dot(X,self.theta[1:])
        return self.sigmoid(prob)
    
    def predict(self, X):
        return np.round(self.predict_proba(X))

        
    def sigmoid(self, z):
        return 1/(1+ np.exp(-z))

    def get_params(self):
        try:
            params = dict()
            params['intercept'] = self.theta[0]
            params['coefficient'] = self.theta[1:]
            return params
        except:
            raise Exception('Fit the model first!')

    def plot_decision_boundary(self, X, y):
        cMap = mcolors.ListedColormap(["#6b76e8", "#c775d1"])
        cMapa = mcolors.ListedColormap(["#c775d1", "#6b76e8"])
        # X_new = pd.DataFrame(X)
        # y_new = pd.Series(y)
        # X_cord = X_new.columns[0]
        # y_cord = X_new.columns[1]
        # x_min = X_new[:, 0].min() - .5
        # x_max = X_new[:, 0].max() + .5
        # y_min = X_new[:, 1].min() - .5
        # y_max = X_new[:, 1].max() + .5
        # h = 0.02 # mesh size
        xx, yy = numpy.mgrid[-5:5:.01, -5:5:.01]
        clf = LogisticRegression().fit(X, y)
        grid = numpy.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict_proba(grid).reshape(xx.shape)
        # Z = Z.reshape(xx.shape)
        f, ax = plt.subplots(figsize=(8, 6))
        # f, ax = plt.subplots(figsize=(8, 6))
        ax.contour(xx, yy, Z, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
        ax.scatter(X.iloc[100:,0], X.iloc[100:, 1], c=y.iloc[100:], s=50, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)
        ax.set(aspect="equal", xlim=(-5, 5), ylim=(-5, 5), xlabel="$X_1$", ylabel="$X_2$")
        # plt.axis('off')
        # plt.pcolormesh(xx, yy, Z, cmap=cMap)
        # plt.scatter(X_new[X_cord], X_new[y_cord], c = y, cmap=cMapa, marker = "o",edgecolors='k', alpha=0.6)
        # plt.xlim(xx.min(), xx.max())
        # plt.ylim(yy.min(), yy.max())
        plt.plot()
        plt.show()
    
    def get_theta(self):
        return self.theta

    def l1_fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        X = np.array(X).astype(np.float64)
        y = np.array(y).reshape(-1,1).astype(np.float64)
        self.theta = np.zeros(X.shape[1]).reshape(-1,1)
        def loss(theta,X,y):
            y_hat = self.sigmoid(np.dot(X,theta))
            y_hat = np.squeeze(y_hat)
            y = np.squeeze(y)
            error = -np.sum(y.dot(np.log10(y_hat))+(1-y).dot(np.log10(1-y_hat)))
            error = error/X.shape[0]
            error += self.l1_coef/(2*X.shape[0])*np.sum(np.abs(theta))
            return error
        delta = grad(loss)

        for _ in range(self.max_iter):
            gradient = delta(self.theta,X,y)
            self.theta -= self.learning_rate*gradient
        return self

    def l2_fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        X = np.array(X).astype(np.float64)
        y = np.array(y).reshape(-1,1).astype(np.float64)
        self.theta = np.zeros(X.shape[1]).reshape(-1,1)
        def loss(theta,X,y):
            y_hat = self.sigmoid(np.dot(X,theta))
            y_hat = np.squeeze(y_hat)
            y = np.squeeze(y)
            error = -np.sum(y.dot(np.log10(y_hat))+(1-y).dot(np.log10(1-y_hat)))
            error = error/X.shape[0]
            error += self.l2_coef/(2*X.shape[0])*np.sum(np.square(theta))
            return error
        delta = grad(loss)

        for _ in range(self.max_iter):
            gradient = delta(self.theta,X,y)
            self.theta -= self.learning_rate*gradient
        return self