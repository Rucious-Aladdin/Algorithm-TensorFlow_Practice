def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x
from numerical_diff import numerical_diff_central
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
plt.show()

print(numerical_diff_central(function_1, 10))
