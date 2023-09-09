from MNIST_DataAugmentation import get_augmentation
import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter7")
from mnist import load_mnist
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cnnclass import SimpleConvNet
import time
from sklearn.metrics import accuracy_score
from Conv_multi import ConvMulti

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

Deep_Conv_acc = []
CONV_acc = []

model = SimpleConvNet()
model.load_params("Conv_MNIST_after_tuning.pkl")
model.save_model("test_SIZE.pkl")    

for i in range(6):
    ## Convolutional Network
    (_, _), (x_test, t_test) =\
            load_mnist(flatten=False)
    x_test = get_augmentation(x_test, shift_vertical=i, shift_horizontal=i)

    model = SimpleConvNet()
    model.load_params("Conv_MNIST.pkl")
    
    acc = model.accuracy(x_test, t_test)
    print(model.loss(x_test, t_test))
    CONV_acc.append(acc)
    
    ##Deep CNN
    Deep_model = ConvMulti()
    Deep_model.load_params("Conv_MNIST_MultiLayer_after_tuning.pkl")
    
    acc = Deep_model.accuracy(x_test, t_test)
    print(Deep_model.loss(x_test, t_test))
    Deep_Conv_acc.append(acc)

print(Deep_Conv_acc)
print(CONV_acc)

# 그래프 크기 및 서브플롯 설정
experiments = range(len(CONV_acc))
x_positions = np.arange(len(experiments))
# 막대 너비 설정
bar_width = 0.4
plt.bar(x_positions - bar_width/4, Deep_Conv_acc, width=bar_width/2, label="Deep CNN", color='green')
plt.bar(x_positions + bar_width/4, CONV_acc, width=bar_width/2, label="CNN", color='red')
plt.title("Test accuracy with Data Augmentation")
plt.xlabel("Max Translation Steps")
plt.ylabel("Test Accuracy")
plt.xticks(x_positions, experiments)
plt.legend()
plt.grid(True)
plt.show()
