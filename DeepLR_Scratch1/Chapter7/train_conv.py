# coding: utf-8
import numpy as np
import sys, os
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1")
import matplotlib.pyplot as plt
from mnist import load_mnist
from cnnclass import SimpleConvNet
from common.trainer import Trainer
import pickle
import time
# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

x_val = x_train[50000:]
t_val = t_train[50000:]
x_train = x_train[:50000]
t_train = t_train[:50000]


# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]
print(x_train.shape)
print(t_train.shape)
print(x_val.shape)
print(t_val.shape)
print(x_test.shape)
print(t_test.shape)

"""
x = x_train[0].reshape(28, -1)
plt.title("Original Image")
plt.imshow(x, cmap="gray")
plt.show()
"""
max_epochs = 1
batch_size = 100
time_list = [0, 0, 0, 0, 0, 0, 0]
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, \
                            'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_val, t_val, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
x = time.perf_counter()
trainer.train(time_list)
x = time.perf_counter() - x
print("training Time: " + str(x))
print("train acc list: " + str(trainer.train_acc_list))
print("val acc list: " + str(trainer.val_acc_list))
print("val_loss_list: " + str(trainer.val_loss_list))

train_loss_avg =[]
a = trainer.train_loss_list[:]
for i in range(int(len(a) / (x_train.shape[0] / batch_size))):
    temp = 0
    for _ in range(int(x_train.shape[0] / batch_size)):
        temp += a[0]
        a.pop(0)
    train_loss_avg.append(temp / (x_train.shape[0] / batch_size))
print("train_loss_avg: " + str(train_loss_avg))
            

print(np.array(time_list) / 600 * 1000)

# 매개변수 보존
network.save_params("Conv_MNIST_after_Tuning.pkl")

# 그래프 그리기
plt.subplot(131)
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, label='train')
plt.plot(x, trainer.val_acc_list, label='val')
plt.legend(loc ="lower right")
plt.title("Train, Val Accuracy(MNIST)")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.grid(True)

plt.subplot(132)
plt.plot(x, train_loss_avg, marker="o", label="train")
plt.plot(x, trainer.val_loss_list, marker="s", label="val")
plt.legend(loc ="upper left")
plt.title("Train, Val Loss(MNIST)")
plt.xlabel("1 Epoch per 500 iters")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(133)
plt.plot(x, train_loss_avg, label="train")
plt.plot(x, trainer.val_loss_list, label="val")
plt.legend(loc ="upper left")
plt.title("Train, Val Loss(MNIST)-ylim(0, 0.5)")
plt.xlabel("1 Epoch per 500 iters")
plt.ylabel("Loss")
plt.ylim(0, 0.5)
plt.grid(True)


plt.show()
"""
실험결과

train loss:2.5114318356973007e-05
train loss:6.005129667399216e-06
train loss:3.838864604186328e-05
=============== Final Test Accuracy ===============
test acc:0.9889
training Time: 5660.880275011063
train acc list: [0.227, 0.971, 0.976, 0.987, 0.984, 0.988, 0.987, 0.99, 0.992, 0.995, 0.992, 0.991, 0.998, 0.998, 0.999, 0.998, 1.0, 0.999, 0.999, 0.999, 0.999, 0.997, 1.0, 1.0, 0.999, 1.0, 0.999, 1.0, 1.0, 0.998]
test ac list: [0.258, 0.966, 0.973, 0.987, 0.988, 0.975, 0.982, 0.983, 0.986, 0.988, 0.984, 0.988, 0.988, 0.988, 0.99, 0.983, 0.988, 0.989, 0.989, 0.989, 0.989, 0.987, 0.984, 0.991, 0.985, 0.986, 0.988, 0.99, 0.988, 0.985]
modelSize: 41345675
"""

"""
backpropagation 소요시간
[2.98091686e-05 6.55886641e-05 7.25066660e-05 7.02827717e-03 2.95898000e-02 4.78623000e-03 4.71581980e-02]

LossLayer: 2.9810e-05

Affine: 6.33886e-05
Relu: 7.2507e-05
Affine: 7.0283e-03

Pooling: 2.95903-02
Relu: 4.7862e-03
Convolution: 4.7158e-02 

**Convolution Layer 소요시간
[0.00565378 0.00900191 0.03213117]
def backward(self, dout, time_list):
    #ConvolutionLayer = time.perf_counter()

    # 1. dout matrix를 reshaping하는 과정
    temp = time.perf_counter()
    FN, C, FH, FW = self.W.shape
    dout = dout.transpose(0,2,3,1).reshape(-1, FN)
    time_list[0] += (time.perf_counter() - temp)
    
    # 2. Convolution Layer의 backward 행렬곱
    temp = time.perf_counter()
    self.db = np.sum(dout, axis=0)
    self.dW = np.dot(self.col.T, dout)
    self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
    time_list[1] += (time.perf_counter() - temp)
    
    
    # 3. Col2im과정
    temp = time.perf_counter()
    dcol = np.dot(dout, self.col_W.T)
    dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
    time_list[2] += (time.perf_counter() - temp)
    
    #print("ConvolutionLayer: " + str(time.perf_counter() - ConvolutionLayer))
    return dx


** POOling Layer 소요시간
[0.01016522 0.01880755]
 # 1. dout matrix를 flatten후 reshaping 하는 과정
        temp = time.perf_counter()
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        time_list[0] += (time.perf_counter() - temp)
        
        # 2. dcol matrix의 col2im과정(2차원 matrix -> 4차원 matrix)
        temp = time.perf_counter()
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        time_list[1] += (time.perf_counter() - temp)
"""

"""
EPOCH = 30
final Experiment Acc
train loss:0.0003337819333652563
train loss:0.00017951295300113062
train loss:5.112856140299443e-05
train loss:6.165220276961943e-05
train loss:0.000572649597240862
train loss:1.8084044060814813e-05
=============== Final Test Accuracy ===============
test acc:0.9892
training Time: 5432.356180700008
train acc list: [0.195, 0.959, 0.975, 0.987, 0.987, 0.985, 0.99, 0.988, 0.99, 0.992, 0.995, 0.996, 0.996, 0.997, 0.993, 0.999, 1.0, 0.999, 1.0, 0.999, 1.0, 1.0, 0.995, 1.0, 0.997, 1.0, 0.999, 1.0, 1.0, 1.0]
val acc list: [0.236, 0.955, 0.971, 0.973, 0.978, 0.978, 0.982, 0.98, 0.981, 0.981, 0.978, 0.986, 0.984, 0.987, 0.982, 0.984, 0.982, 0.983, 0.981, 0.984, 0.983, 0.984, 0.98, 0.985, 0.984, 0.988, 0.982, 0.982, 0.986, 0.984]
val_loss_list: 30
train_loss_avg: [0.36683817494009746, 0.10422187210985717, 0.060723267140663166, 0.04034749142296333, 0.034080903803871826, 
0.02348031203430466, 0.01852256297396977, 0.01569769191318941, 0.011362519897081707, 0.011354936072799606, 0.008427935165909298, 0.006329494838957734, 0.005784896092346186, 0.004393593573921407, 0.004036893561098676, 0.004148512734185015, 0.0030860738535286947, 0.0015191819236720849, 0.002277658844234246, 0.0022229665701160614, 0.0015223714272329905, 0.0010191278148718316, 0.001319574265859313, 0.0010747325658724818, 0.0007999260020427835, 0.001370893654619555, 0.0010479052028347085, 0.0009804986426926834, 0.0008418980877025758, 0.0007428659769212584]

"""

"""

EPOCH = 13

train loss:0.0012340854481632055
train loss:0.0005117182272120778
train loss:0.0008198496698924361
=============== Final Test Accuracy ===============
test acc:0.9889
training Time: 2638.298908099998
train acc list: [0.147, 0.96, 0.977, 0.978, 0.986, 0.989, 0.983, 0.992, 0.995, 0.995, 0.993, 0.996, 0.996]
val acc list: [0.122, 0.954, 0.977, 0.972, 0.973, 0.982, 0.98, 0.978, 0.984, 0.984, 0.98, 0.985, 0.98]
val_loss_list: [2.300595435152693, 0.1427637557552527, 0.08438181430811437, 0.07309801498080262, 0.0670719186520685, 0.057456776376412236, 0.05933522100118502, 0.05123921463126035, 0.05406224159858151, 0.05112712718902579, 0.05377431752056522, 0.054796279969642855, 0.05373823693498434]
train_loss_avg: [0.359169273287758, 0.10350265196990882, 0.062145720031923465, 0.04198976712312038, 0.031696860672152175, 0.024354469617225493, 0.01994127064573978, 0.016864771272328296, 0.010848534204804871, 0.009772399377805429, 0.007950110980130536, 0.006248065010015825, 0.005798441314816587]


"""