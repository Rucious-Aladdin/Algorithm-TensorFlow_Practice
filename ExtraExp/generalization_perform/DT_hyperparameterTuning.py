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

# DecisionTreeClassifier 모델 생성
clf = DecisionTreeClassifier()
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=False)
x_train = x_train
t_train = t_train
x_test = x_test
t_test = t_test
test_acc = []
time_list = []

param_dist = {
    "max_depth": list(range(5, 31, 5)),
    "min_samples_split": list(range(2, 21)),
    "min_samples_leaf": list(range(1, 21))
}
num_samples = 10
random_hyperparameters = {
    "max_depth": np.random.choice(param_dist["max_depth"], num_samples),
    "min_samples_split": np.random.choice(param_dist["min_samples_split"], num_samples),
    "min_samples_leaf": np.random.choice(param_dist["min_samples_leaf"], num_samples)
}

test_acc = []
time_list = []
size_list = []
train_acc = []

for i, hyperparameters in enumerate(zip(random_hyperparameters["max_depth"],
                                        random_hyperparameters["min_samples_split"],
                                        random_hyperparameters["min_samples_leaf"])):
    model = DecisionTreeClassifier(
        max_depth=hyperparameters[0], 
        min_samples_split=hyperparameters[1],
        min_samples_leaf=hyperparameters[2]
        )
    
    start = time.time()
    model.fit(x_train, t_train)
    time_list.append(time.time() - start)
    
    y_pred = model.predict(x_test)
    test_accuracy = accuracy_score(t_test, y_pred)
    test_acc.append(test_accuracy)
    
    y_train_pred = model.predict(x_train)
    train_accuracy = accuracy_score(t_train, y_train_pred)
    train_acc.append(train_accuracy)
    
    # 모델 크기 출력
    model_size = sys.getsizeof(pickle.dumps(model))
    size_list.append(model_size)
    
    model.save_model("decisionTree" + str(i) + ".pkl")

print(test_acc)
print(train_acc)
print(time_list)
print(size_list)

# 그래프 크기 및 서브플롯 설정
plt.figure(figsize=(15, 5))
experiments = range(len(test_acc))

x_positions = np.arange(len(experiments))
# 막대 너비 설정
bar_width = 0.4
plt.subplot(131)
plt.bar(x_positions - bar_width/2, train_acc, width=bar_width, label="Train Accuracy", color='blue')
plt.bar(x_positions + bar_width/2, test_acc, width=bar_width, label="Test Accuracy", color='green')
plt.title("Train / Test Accuracy")
plt.xlabel("Experiment")
plt.ylabel("Accuracy")
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

for i, hyperparameters in enumerate(zip(random_hyperparameters["max_depth"],
                                        random_hyperparameters["min_samples_split"],
                                        random_hyperparameters["min_samples_leaf"])):
    print(str(hyperparameters[0]) + " " + str(hyperparameters[1])+ " " + str(hyperparameters[2]))
    
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