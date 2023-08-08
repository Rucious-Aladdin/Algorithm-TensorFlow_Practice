import numpy as np
import sys, os
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter4")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter3")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
from common.util import im2col

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) // self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) // self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad)
        print(col.shape)
        col_W = self.W.reshape(FN, -1).T  # 필터 전개
        print(col_W.shape)
        out = np.dot(col, col_W) + self.b
        print(out.shape)
        
        out = out.reshape(N, out_h, out_w, -1)
        print(out.shape)
        out = out.transpose(0, 3, 1, 2)
        print(out.shape)
        return out
    
test_input = np.random.rand(10, 3, 7, 7)
test_kernel = np.random.rand(2, 3, 5, 5)
b = np.random.rand(1, 2)
model = Convolution(test_kernel, b)
model.forward(test_input)
