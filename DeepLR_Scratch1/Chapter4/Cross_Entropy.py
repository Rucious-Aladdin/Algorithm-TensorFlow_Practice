import numpy as np
# 교차 엔트로피 손실함수

# if t = [0.1, 0.1. 0.8] (predict)
# if y = [0.0, 0.0, 1.0] (label)
#... 이면 교차 엔트로피 loss는 -(0 * log 0.1 + 0 * log 0.1 + 1.0 * log 0.8)이고, 
# label이 들어간 부분에 들어간 숫자가 1에 가까울수록 손실이 작아짐.
# 0에 가까우면, 손실이 더욱 커지는 구조..


def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
t = [0, 0, 1, 0, 0, 0, 0, 0 ,0, 0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(cross_entropy(np.array(y1), np.array(t)))
print(cross_entropy(np.array(y2), np.array(t)))