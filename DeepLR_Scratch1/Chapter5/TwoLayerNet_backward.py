import numpy as np
import os, sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter4")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter3")
from Cross_Entropy_batch import cross_entropy_error
from softmax import softmax
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
import pickle
import time
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std = 0.01):
        zr = np.zeros
        rn = np.random.randn
        on = np.ones
        self.params = {}
        self.params["W1"] = 0.0001 * rn(input_size, hidden_size).astype("float64")
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = 0.0001 * rn(hidden_size, output_size).astype("float64")
        self.params["b2"] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers["Affine1"] = \
            Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = \
            Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
        
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
           
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])
        
        return grads
        
    def gradient(self, x, t, time_list):
        self.loss(x, t)
        dout = 1
        temp = time.perf_counter()
        dout = self.lastLayer.backward(dout)
        time_list[0] += (time.perf_counter() - temp)
        layers = list(self.layers.values())
        layers.reverse()
        for i, layer in enumerate(layers):
            temp = time.perf_counter()
            dout = layer.backward(dout)
            time_list[i+1] += (time.perf_counter() - temp)
        
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        
        return grads
    
    def save_model(model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

    # 모델 로드
    def load_model(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model