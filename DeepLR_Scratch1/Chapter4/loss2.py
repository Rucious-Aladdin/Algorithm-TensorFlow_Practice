import sys, os
sys.path.append(os.pardir)
sys.path.append("c:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
from mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)
    
print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask) # 0이상 60000미만 숫자중 무작위 출력
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]