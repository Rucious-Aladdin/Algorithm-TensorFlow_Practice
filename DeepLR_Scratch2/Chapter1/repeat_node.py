import numpy as np

D, N = 8, 7

x = np.random.randn(1, D) # 열벡터 x를 정의
y = np.repeat(x, N, axis = 0) # 0축(행방향)을 따라서 복제

dy = np.random.randn(N, D)
dx = np.sum(dy, axis = 0, keepdims=True)

print(dx.shape)
print(dx)