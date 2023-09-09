import numpy as np
import sys, os
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1")
from common.util import im2col
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
import pickle
import time

class SimpleConvNet:
    def __init__(self, input_dim = (1, 28, 28),
                 conv_param = {"filter_num":30, "filter_size":5,
                               "pad":0, "stride":1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / \
                            filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * 
                               (conv_output_size / 2))
        
        self.params = {}
        self.params["W1"] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0],
                                            filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std * \
                            np.random.randn(pool_output_size, 
                                            hidden_size)
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)
        
        
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(self.params["W1"],
                                           self.params["b1"], 
                                           conv_param["stride"],
                                           conv_param["pad"])
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W2"],
                                        self.params["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W3"],
                                        self.params["b3"])
        self.last_layer = SoftmaxWithLoss()
        
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def gradient(self, x, t, time_list=None):
        temp = time.perf_counter()
        self.loss(x, t)
        time_list[0] += (time.perf_counter() - temp)
        
        temp = time.perf_counter()
        dout = 1
        dout = self.last_layer.backward(dout)
        #print("SoftmaxWithLossLayer Bacpropagation: " + str(time.perf_counter() - loss_time))
        #time_list[0] += (time.perf_counter() - loss_time)
        layers = list(self.layers.values())
        layers.reverse()
        for i, layer in enumerate(layers):
            #temp = time.perf_counter()
            dout = layer.backward(dout)
            #time_list[i+1] += (time.perf_counter() - temp)
        time_list[1] += (time.perf_counter() - temp)
        
        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Affine1"].dW
        grads["b2"] = self.layers["Affine1"].db
        grads["W3"] = self.layers["Affine2"].dW
        grads["b3"] = self.layers["Affine2"].db
        
        return grads
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
    
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]
    
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
            
    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads
    
    def save_model(model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)