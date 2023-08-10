import numpy as np

D, N = 8, 7
## forward

x = np.random.randn(N, D)
y = np.sum(x, axis = 0, keepdims=True)
print(y)
## backward

dy = np.random.randn(1, D)
dx = np.repeat(dy, N, axis=0)
print(dx) # 그대로 복제된 모습