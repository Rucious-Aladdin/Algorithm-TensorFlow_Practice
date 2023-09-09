from decisionTree import DecisionTreeClassifier
import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
from mnist import load_mnist
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
    [0.8799, 0.8811, 0.8801, 0.8764, 0.8851, 0.8719, 0.8725, 0.8701, 0.8752, 0.8851]
[0.9162833333333333, 0.9218333333333333, 0.9161166666666667, 0.9114333333333333, 0.9586, 0.9083, 0.9044833333333333, 
0.8984833333333333, 0.9061333333333333, 0.9586]
[3254.3471863269806, 2962.4883716106415, 2900.0751056671143, 2904.8168256282806, 3165.255027294159, 2446.88712477684, 2780.5441834926605, 2416.5511281490326, 2813.705292224884, 3146.8785054683685]
[48004, 52027, 46700, 44167, 80202, 33426, 39803, 28672, 40831, 80202]
"""
val_acc = [0.8799, 0.8811, 0.8801, 0.8764, 0.8851, 0.8719, 0.8725, 0.8701, 0.8752, 0.8851]
print(val_acc)
train_acc = [0.9162833333333333, 0.9218333333333333, 0.9161166666666667, 0.9114333333333333, 0.9586, 0.9083, 0.9044833333333333, 
0.8984833333333333, 0.9061333333333333, 0.9586]
print(train_acc)
time_list = [3254.3471863269806, 2962.4883716106415, 2900.0751056671143, 2904.8168256282806, 3165.255027294159, 2446.88712477684, 2780.5441834926605, 2416.5511281490326, 2813.705292224884, 3146.8785054683685]
print(time_list)
size_list = [48004, 52027, 46700, 44167, 80202, 33426, 39803, 28672, 40831, 80202]
print(size_list)

# 그래프 크기 및 서브플롯 설정
plt.figure(figsize=(15, 5))
experiments = range(len(val_acc))

x_positions = np.arange(len(experiments))
# 막대 너비 설정
bar_width = 0.4
plt.subplot(131)
plt.bar(x_positions - bar_width/2, train_acc, width=bar_width, label="Train Accuracy", color='blue')
plt.bar(x_positions + bar_width/2, val_acc, width=bar_width, label="Val. Accuracy", color='green')
plt.title("Train / Val Accuracy")
plt.xlabel("Experiment")
plt.ylabel("Accuracy")
plt.ylim(0.5, 1.0)
plt.xticks(x_positions, experiments)
plt.legend()
plt.grid(True)

# 학습 시간 서브플롯
plt.subplot(132)
plt.bar(experiments, time_list, width=0.4, color="purple")
plt.title("Training Time")
plt.xlabel("Experiment")
plt.ylabel("Training Time (s)")
plt.grid(True)

# 모델 크기 서브플롯
plt.subplot(133)
plt.bar(experiments, size_list, width=0.4, color="pink")
plt.title("Model Size")
plt.xlabel("Experiment")
plt.ylabel("Model Size (bytes)")
plt.grid(True)

# 서브플롯 간 간격 조정
plt.tight_layout()

# 그래프 표시
plt.show()


"""
    [0.8799, 0.8811, 0.8801, 0.8764, 0.8851, 0.8719, 0.8725, 0.8701, 0.8752, 0.8851]
[0.9162833333333333, 0.9218333333333333, 0.9161166666666667, 0.9114333333333333, 0.9586, 0.9083, 0.9044833333333333, 
0.8984833333333333, 0.9061333333333333, 0.9586]
[3254.3471863269806, 2962.4883716106415, 2900.0751056671143, 2904.8168256282806, 3165.255027294159, 2446.88712477684, 2780.5441834926605, 2416.5511281490326, 2813.705292224884, 3146.8785054683685]
[48004, 52027, 46700, 44167, 80202, 33426, 39803, 28672, 40831, 80202]
25 13 14
30 11 12
15 20 14
20 13 16
25 6 5
10 7 5
30 16 19
10 10 11
20 2 18
20 9 5
    """