import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3, 0], [0, 6]])

print(A * B) # element-wise product
print(np.matmul(A, B)) # matrix multiply

print(A)
print(A * 10)