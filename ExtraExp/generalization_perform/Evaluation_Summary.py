import matplotlib.pyplot as plt
import numpy as np

# 항목 정의
categories = ['DT', 'FFNN', 'CNN']

# train, test, train-test, axis = 0
data = np.array([[0.9586, 0.8851, 0.9586 - 0.8851], [0.99424, 0.9723, 0.99424 - 0.9723], [0.9960, 0.9889, 0.9960 - 0.9889]])
# model_size & Training Time
size_list = np.array([79, 989, 3391])
training_time_list = np.array([ 5.7406, 202.9333])
print(data)
# 그래프 그리기
width = 0.07  # 막대 너비
x = np.arange(len(categories))

fig, ax = plt.subplots()

ax.bar(x + 0 * width, data[:, 0], width, label="Train Accuracy")
ax.bar(x + 1 * width, data[:, 1], width, label="Test Accuracy")
ax.bar(x + 2 * width, data[:, 2], width, label="Train - Test")

ax.set_xlabel('Categories')
ax.set_ylabel('Accuracy')
ax.set_title('Generalization Performance Compare')
ax.set_xticks(x + width)
ax.set_xticklabels(categories)
ax.legend(loc="center")
plt.show()

plt.subplot(121)
bar_width = 0.4
# 막대 그래프 그리기
plt.bar(categories, size_list, width=bar_width)
# x축 눈금(label) 설정
plt.xticks(categories)
# 그래프 제목과 라벨 추가
plt.title('Model Size Evaluation')
plt.xlabel('Categories')
plt.ylabel('Size(KBytes)')
plt.legend(loc="center")
# 그래프 표시

plt.subplot(122)
categories.pop(0)
bar_width = 0.4
# 막대 그래프 그리기
plt.bar(categories, training_time_list, width=bar_width)
# x축 눈금(label) 설정
plt.xticks(categories)
# 그래프 제목과 라벨 추가
plt.title('Training Time per 1 Epoch')
plt.xlabel('Categories')
plt.ylabel('Time(s)')

# 그래프 표시


plt.show()