import numpy as np

def relu(x):
    return np.maximum(0, x)

A = np.array([-1, 2, 3, 4])
print(relu(A)) #-1이 0으로 바뀌는 것을 볼 수 있음.
