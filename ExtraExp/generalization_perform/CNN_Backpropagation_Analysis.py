import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 12})

#Label = ["Loss", "Affine2", "Relu2", "Affine1", "Pooling", "Relu1", "Convolution"]
#a = [2.98091686e-05, 6.55886641e-05, 7.25066660e-05, 7.02827717e-03, 2.95898000e-02, 4.78623000e-03, 4.71581980e-02]
a = [6.956445,  7.503425,  1.10312083]
Label = ["Forward", "Backward", "Update"]
a.reverse()
Label.reverse()

a = np.array(a)
print(a)
print(np.sum(a))
# 고유한 레이블을 찾아내고 각 레이블에 대한 색상을 생성합니다.
unique_labels = list(set(Label))
#colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
colors = ["blue", "brown", "green"]
colors.reverse()
# 레이블과 색상을 매핑합니다.
label_color_mapping = {label: color for label, color in zip(unique_labels, colors)}

# 각 레이블에 해당하는 데이터를 추출합니다.
data_by_label = {label: [] for label in unique_labels}
for label, value in zip(Label, a):
    data_by_label[label].append(value)

# 순서를 유지한 레이블 및 데이터 리스트를 생성합니다.
sorted_labels = [label for label in Label]
sorted_data = [data_by_label[label] for label in sorted_labels]


# 막대 그래프를 그립니다.
plt.figure(figsize=(10, 6))

plt.subplot(121)
plt.bar(sorted_labels, [np.mean(data) for data in sorted_data], color=[label_color_mapping[label] for label in sorted_labels])
plt.xlabel('Layer')
plt.ylabel('time (ms)')
plt.title('Backpropagation Time Analysis (average of 1 epoch)')
plt.legend()

# 데이터를 정렬합니다.
sorted_data, sorted_labels = zip(*sorted(zip(a, Label), reverse=True))

# 합계를 계산합니다.
total = sum(sorted_data)

# 비율이 일정 값 이상인 항목만 표시합니다.
threshold = 0.01  # 원하는 임계값을 설정하세요 (예: 1%)
filtered_data = [data if data / total >= threshold else 0 for data in sorted_data]
filtered_labels = [label if data / total >= threshold else 'etc' for data, label in zip(sorted_data, sorted_labels)]

plt.subplot(122)
# 원형 그래프를 그립니다.
plt.pie(filtered_data, labels=filtered_labels, colors=[label_color_mapping[label] for label in sorted_labels], autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Training time')

plt.show()