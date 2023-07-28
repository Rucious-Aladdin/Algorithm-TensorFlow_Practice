import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from Perceptron import Perceptron
from Iris import plot_decision_regions


s = os.path.join("https://archive.ics.uci.edu", "ml", "machine-learning-databases", "iris", "iris.data", )
s = s.replace("\\", "/")
print(s)
df = pd.read_csv(s, header = None, encoding = "utf-8")
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1) # return 조건에 맞는 label의 index list

X = df.iloc[0:100, [0, 2]].values # integer location, values -> np types return
print(X.shape) # (100, 2)

class AdalineGD(object):
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
        self.cost_ = []
            
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.01, 1, -1)
    
#fig, ax = plt.subplots(nrows =1, ncols = 2, figsize = (10, 4))
"""
ada1 = AdalineGD(n_iter=10, eta = 0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker="o")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log(SSE loss)")
ax[0].set_title("AdaLine - LR = 0.01")

ada2 = AdalineGD(n_iter=10, eta = 0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker="o")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("SSE loss")
ax[1].set_title("AdaLine - LR = 0.0001")
plt.show()
"""
X_std = np.copy(X) # 복사

## 표준화 -> mean = 0, std = 1 <-> (unit Variance)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(n_iter = 15, eta = 0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title("Adaline - GD")
plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = "o")
plt.xlabel("Epochs")
plt.ylabel("SSE")
plt.tight_layout()
plt.show()

