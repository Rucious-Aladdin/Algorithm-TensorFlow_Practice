import numpy as np

if __name__ == "__main__":
    T, H = 5, 4
    hs = np.random.randn(T, H)
    a = np.array([0.8, 0.1, 0.03, 0.05, 0.02])

    ar = a.reshape(5, 1).repeat(4, axis=1)
    ar2 = a.reshape(5, 1).repeat(4, axis=0)
    print("ar: ", ar.shape)
    print("ar2.shape", ar2.shape)

    t = hs * ar
    print(t.shape)

    c = np.sum(t, axis=0)
    print(c.shape)
    
class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
    
    def forward(self, hs, a):
        N, T, H = hs.shape
        
        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)
        
        self.cache = (hs, ar)
        return c
    
    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2)
        
        return dhs, da