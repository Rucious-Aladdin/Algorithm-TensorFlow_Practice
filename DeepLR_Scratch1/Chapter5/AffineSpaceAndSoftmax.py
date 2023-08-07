import numpy as np
import os, sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter4")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter3")
from Cross_Entropy_batch import cross_entropy_error
from softmax import softmax

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(self.W, x) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        
        return dx
        
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t =t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.x)
        return self.loss
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

if __name__ == "__main__":
    FORWARD = False
    SOFT = True
    if FORWARD:
        X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
        B = np.array([1, 2, 3])
        
        print(X_dot_W)
        print(X_dot_W + B)
        
        dY = np.array([[1, 2, 3], [4, 5, 6]])
        dB = np.sum(dY, axis=0)
        print(dB)
    
    if SOFT:
        pass