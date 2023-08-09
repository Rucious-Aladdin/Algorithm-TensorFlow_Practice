import numpy as np

x = np.array([1, 2, 3])
print(x.__class__)
print(x.shape)
print(x.ndim)

W = np.array([[1, 2, 3], [4, 5, 6]])
print(W.shape)
print(W.ndim)

W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])

print(W + X)
print(W * X)

