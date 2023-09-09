# coding: utf-8
import numpy as np
import sys, os
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1")
import matplotlib.pyplot as plt
from mnist import load_mnist
from Conv_multi import ConvMulti
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
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]
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
max_epochs = 15
batch_size = 100
time_list = [0, 0, 0, 0, 0, 0, 0]
network = ConvMulti(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, \
                             "filter2_size":3, "filter2_num":30, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_val, t_val, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
x = time.perf_counter()
trainer.train()
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
network.save_params("Conv_MNIST_MultiLayer_after_tuning.pkl")

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
train loss:0.00015074414346037904
train loss:0.0006939258368967216
train loss:0.0004983929932053668
=============== Final Test Accuracy ===============
test acc:0.99
training Time: 8269.770971699967
train acc list: [0.164, 0.961, 0.975, 0.979, 0.986, 0.988, 0.985, 0.99, 0.989, 0.989, 0.99, 0.994, 0.9 0.995, 0.993, 0.997, 0.997, 0.998, 0.996, 0.995, 1.0, 1.0, 0.997, 0.993, 0.998, 1.0, 1.0, 1.0]
val acc list: [0.191, 0.957, 0.974, 0.983, 0.981, 0.981, 0.98, 0.982, 0.983, 0.989, 0.983, 0.988, 0.99
0.988, 0.988, 0.99, 0.99, 0.988, 0.987, 0.987, 0.986, 0.991, 0.986, 0.983, 0.984, 0.989, 0.987, 0.995]
6, 0.060464380023815816, 0.045288058681121755, 0.043074309399644566, 0.04785831176899484, 0.04754271917483145, 0.044781334985104004, 0.0415374411797031, 0.046981991144370745, 0.0593935301520681, 0.05859259802469228, 0.05085349773880895, 0.054810514669350656, 0.06839513856618705, 0.060718043390507945, 0.0523132104766074, 0.05601687759937608, 0.05055292089440942]
train_loss_avg: [0.35570635430139386, 0.08700233036926924, 0.06374146157898722, 0.04808990716703988, 0.03931385885476979, 0.034442397772913656, 0.02949178481172859, 0.02560667186681448, 0.023119318163065554, 0.0190276002317354, 
0.016622934557851667, 0.01527120997469378, 0.012984618446055394, 0.01209662478782989, 0.011080426062616474, 0.009317951522334278, 0.00809718677126011, 0.007743537452751541, 0.007483148037990095, 0.0063882224676769595, 0.0052403870385503174, 0.005284097003206419, 0.005449559590820161, 0.0037391838276445025, 0.0027734598966517392, 0.0033345501479816343, 0.004369095549858771, 0.003124361719562354, 0.0032594946683758663, 0.0027434996976942873]
[0. 0. 0. 0. 0. 0. 0.]

"""