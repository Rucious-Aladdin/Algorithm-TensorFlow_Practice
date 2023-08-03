import numpy as np
import matplotlib.pyplot as plt

# 1. np.where을 이용한 구현
def step_function(x):
    return np.where(x > 0, 1, 0)

# 2. y.astype를 이용한 구현
def step_function2(x):
    y = x > 0
    #y는.. [True, False, True]가됨.
    return y.astype(np.int32)
    # int32, int64로 자료형의 크기를 명시해야 쓸 수 있는듯?
    # 넘파이 배열의 자료형을 변환.. True, False -> 1, 0

x = np.array([1.0, -1.0, 3.0])
print(step_function(x))
print(step_function2(x))

def step_function3(x):
    return np.array(x > 0, dtype=np.int32)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function3(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()