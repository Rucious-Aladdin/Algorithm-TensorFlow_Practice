from MNIST_DataAugmentation import get_augmentation
from decisionTree import DecisionTreeClassifier
import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter7")
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\Chapter5")
from mnist import load_mnist
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cnnclass import SimpleConvNet
from decisionTree import DecisionTreeClassifier
import time
from sklearn.metrics import accuracy_score
from TwoLayerNet_backward import TwoLayerNet
from Conv_multi import ConvMulti

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

DT_acc = []
FFNN_acc = []
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
    
    ## DECISION TREE
    (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize=True, one_hot_label=False, flatten=False)
    x_test = get_augmentation(x_test, shift_vertical=i, shift_horizontal=i)
    x_test = x_test.reshape(10000, -1)

    with open("decisionTree4.pkl", "rb") as f:
        model_DT = pickle.load(f)

    y_pred = model_DT.predict(x_test)
    acc = accuracy_score(t_test, y_pred)
    DT_acc.append(acc)
    
    ## FFNN
    """
    x_test = get_augmentation(x_test, shift_vertical=i, shift_horizontal=i)
    x_test = x_test.reshape(10000, -1)
    """
    model_filename = "two_Layer_net_MNIST.pkl"
    model_FFNN = load_model(model_filename)
    
    acc = model_FFNN.accuracy(x_test, t_test)
    print(model_FFNN.loss(x_test, t_test))
    FFNN_acc.append(acc)

print(DT_acc)
print(FFNN_acc)
print(CONV_acc)

# 그래프 크기 및 서브플롯 설정
experiments = range(len(DT_acc))
x_positions = np.arange(len(experiments))
# 막대 너비 설정
bar_width = 0.4
plt.bar(x_positions - bar_width/2, CONV_acc, width=bar_width/3, label="CNN", color='red')
plt.bar(x_positions - bar_width/6, FFNN_acc, width=bar_width/3, label="FFNN", color='green')
plt.bar(x_positions + bar_width/6, DT_acc, width=bar_width/3, label="DT", color='blue')
plt.title("Test accuracy with Data Augmentation")
plt.xlabel("Shift Steps")
plt.ylabel("Test Accuracy")
plt.xticks(x_positions, experiments)
plt.legend()
plt.grid(True)
plt.show()
