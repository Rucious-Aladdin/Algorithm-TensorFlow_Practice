# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from Conv_multi import ConvMulti

def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = ConvMulti()
# 무작위(랜덤) 초기화 후의 가중치
filter_show(network.params['W1'])
filter_show(network.params["W2"])
# 학습된 가중치
network.load_params("Conv_MNIST_MultiLayer.pkl")
filter_show(network.params['W1'])
filter_show(network.params["W2"])