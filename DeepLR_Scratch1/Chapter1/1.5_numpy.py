import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

y = np.array([2.0, 4.0, 6.0])

print(x + y)
print(x - y)
print(x * y) # element-wise product
print(x / y) # element-wise divide

print(np.dot(x, y.T)) #dot product of 2 vectors
print(x / 2.0) # broadcasting.. eigen value를 곱하는 것과 매우 유사.