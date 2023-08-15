import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch2")
import numpy as np
from common.layers import Softmax

N, T, H = 10, 5, 4
hs = np.random.randn(N, T, H)
h = np.random.randn(N, H)
hr = h.reshape(N, 1, H).repeat(T, axis=1)

t = hs * hr
print(t.shape)

s = np.sum(t, axis = 2)
print(s.shape)

softmax = Softmax()
a = softmax.forward(s)
print(a.shape)
print(a)
print(np.sum(a, axis = 1)) # 열축 기준 합이 1이되는지 확인.