import numpy as np

x = np.random.randn(10, 3, 28, 28)

print(x.shape)

print(x[0, 0].shape) # 첫번째 데이터의 첫번째 채널의 데이터에 접근 -> (28, 28)의 2차원 배열
