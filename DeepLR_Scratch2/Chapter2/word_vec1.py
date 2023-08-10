import numpy as np
import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch2")
from common.layers import MatMul

c = np.array([1, 0, 0, 0, 0, 0, 0])
W = np.random.randn(7, 3)
layer = MatMul(W)
h = layer.forward(c)
print(h)
