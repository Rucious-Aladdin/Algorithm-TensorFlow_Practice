import sys, os
import numpy as np
sys.path.append("c:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
from mnist import load_mnist
import pickle
from Sigmoid_Func import sigmoid
from softmax import softmax
#from implement_sum import init_network

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter3\\sample_weight.pkl", "rb") as f:
        network = pickle.load(f) 
    return network

def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

if __name__ == "__main__":
    x, t = get_data()
    network = init_network()
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    batch_size = 100
    accuracy_cnt = 0
    
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis = 1) # 확률이 가장 높은 원소 index
        print(p==t[i:i+batch_size])
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
    x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
    print(x.shape)
    # axis = 0 -> 세로축
    # axis = 1 -> 가로축