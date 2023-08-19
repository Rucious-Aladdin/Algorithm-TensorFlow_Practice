import numpy as np
import matplotlib.pyplot as plt
from collections import *
import math
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out
        return dx
    
class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask]
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

class SseLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = x
        z = (x - t) ** 2
        self.loss = 1 / (2 * len(x)) * np.sum(z)
        return self.loss
        
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size
        
        return dx
            
class one_hidden_net:
    def __init__(self, hidden_size, input_size, output_size, std = 0.1):
        self.params = {}
        
        a1 = math.sqrt(6) / math.sqrt(1 + hidden_size)
        print(a1)
        a2 = math.sqrt(6) / math.sqrt(hidden_size * 1.5)
        print(a2)
        self.params["W1"] = np.random.uniform(-a1, a1, size = (input_size, hidden_size))
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = np.random.uniform(-a2, a2, size = (hidden_size, hidden_size // 2))
        self.params["b2"] = np.zeros(hidden_size // 2)
        self.params["W3"] = np.random.uniform(-a1, a1, size = (hidden_size // 2, output_size))
        self.params["b3"] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers["Affine1"] = \
            Affine(self.params["W1"], self.params["b1"])
        self.layers["Sigmoid"] = Sigmoid()
        self.layers["Affine2"] = \
            Affine(self.params["W2"], self.params["b2"])
        self.layers["Sigmoid"] = Sigmoid()
        self.layers["Affine3"] = \
            Affine(self.params["W3"], self.params["b3"])
        
        self.loss_layer = SseLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.loss_layer.forward(y, t)
    
    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.loss_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        grads["W3"] = self.layers["Affine3"].dW
        grads["b3"] = self.layers["Affine3"].db
        
        return grads
 
def test_func1(x):
    return 0.01 * x * (x - 3) * (x + 3)

def test_func2(x):
    return 0.1 * np.sin(x)
    
def load_data(function, min, max, sample_num = 1000):
    step = (max - min) / sample_num
    x_train = np.arange(min, max,  step)
    t_train = function(x_train)
    
    return x_train, t_train


if __name__ == "__main__":
    hidden_size = 50
    input_size = 1
    output_size = 1
    lr = 0.01
    iters_num = 50000
    
    min = -10
    max = 10
    sample_num = 1000
    batch_size = 20
    train_size = sample_num
    
    model = one_hidden_net(hidden_size, input_size, output_size, std=0.001)
    
    x_train1, t_train1 = load_data(test_func2, min, max, sample_num)
    
    train_loss_list = []
    train_acc_list = []
    
    iter_per_epoch = train_size / batch_size
    
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train1[batch_mask].reshape(batch_size, 1)
        t_batch = t_train1[batch_mask].reshape(batch_size, 1)
        
        grad = model.gradient(x_batch, t_batch)
        
        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            model.params[key] -= lr * grad[key]
        
        loss = model.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_loss = model.loss(x_train1.reshape(len(x_train1), 1), t_train1.reshape(len(t_train1), 1))
            print("iteration: " + str(i)  + " train loss: " + str(train_loss))
    
    plt.plot(x_train1, model.predict(x_train1.reshape(len(x_train1), 1)))
    plt.plot(x_train1, t_train1)
    plt.show()
    
    