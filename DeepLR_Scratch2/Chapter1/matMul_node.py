import numpy as np

class MatMul:
    def __init__(self, W):
        self.params [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        
    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW 
        # 점3개로 이루어진 생략기호 사용, 넘파이배열이 가리키는 메모리 위치고정, 덮어쓰기 수행.   
        
        return dx