import numpy as np
import os, sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter4")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter3")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
from mnist import load_mnist
from TwoLayerNet_backward import TwoLayerNet
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_num = 100000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iter_per_epoch = train_size // batch_size
temp = {}
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #이전 코드에서 바뀐부분 -> (수치미분 -> 역전파 방식)
    grad = network.gradient(x_batch, t_batch)
    if i == iters_num-1:
        temp = grad
        print("W1" + str(network.params["W1"]))
        print("W2" + str(network.params["W2"]))
        print(grad["W1"])
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]
        
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % 60 == 0:
        test_acc = network.accuracy(x_test, t_test)
        train_acc = network.accuracy(x_train, t_train)    
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

x = range(len(train_acc_list))
plt.plot(x, train_acc_list, x, test_acc_list)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("MNIST accuracy")
plt.show()