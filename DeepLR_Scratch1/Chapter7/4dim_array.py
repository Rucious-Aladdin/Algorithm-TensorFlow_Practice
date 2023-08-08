import numpy as np

x = np.random.randn(10, 3, 28, 28)

print(x.shape)

print(x[0, 0].shape) # 3