import numpy as np
a = np.random.randn(3)
print(a.dtype)

b = np.random.randn(3).astype(np.float32)
print(b.dtype)