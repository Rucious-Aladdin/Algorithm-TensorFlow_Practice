import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A * 10)

b = np.array([[10], [20]]) # axis = 1
b2 = np.array([10, 20, 30]) # axis = 0

print(A * b)
print(A * b2)