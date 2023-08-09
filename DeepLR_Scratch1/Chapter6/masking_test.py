import numpy as np

### False * 실수 = 0 즉.. dropout은 false이 부분
x = np.array([1, 2, 3])
y = np.array([True, False, False])

print(x * y)