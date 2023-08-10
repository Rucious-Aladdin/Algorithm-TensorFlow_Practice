import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

x = np.random.randn(10, 2)
h = np.matmul(x, W1) + b1

# b1, b2의 경우 broadcasting 된점 참고!
print(sigmoid(h))
a = sigmoid(h)
s = np.matmul(a, W2) + b2