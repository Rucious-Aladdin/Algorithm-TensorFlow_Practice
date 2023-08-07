#PT내용에 추가
# Xavier, HE의 경우 논문 참조할 것.
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)
    
input_data = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장
exp = [3, 2]

'''
앞쪽 index -> 0: N(0, 1), 1: N(0, 0.01), 2: Xavier, 3: HE
뒤쪽 index -> 0: sigmoid, 1: Relu, 2: tanh

-------N(0, 1)--------
0, 0 -> 값이 0, 1 양 극단에 치우침.
0, 1 -> 값이 0쪽 극단에 치우침.
0, 2 -> 값이 1쪽 극단에 치우침.

-------N(0, 0.01)-----
1, 0 -> 0.5 중앙에 쏠림
1, 1 -> 0 쪽에 쏠림. 첫번째 레이어는 정상적으로 작동. Dying Relu문제로 보임.
1, 2 -> 첫번째 레이어는 0~1사이 에 치우친 왜도가 높은 분포, 뒤쪽은 0에만 몰린 극단적 분포.

------Xavier---------
2, 0 -> 고른 분포를 보임. 뒤쪽 레이어로 갈수록 극단의 값들이 사라지는 경향 있음.
2, 1 -> 0주변 매우 높은 분포, 뒤쪽에도 분포 존재, 마지막 히든레이어로 갈수록 0쪽으로 몰리는 값이 많아지는 경향
2, 2 -> 첫번쨰 레이어 -> 0에서 1까지 높아지는 분포, 두번째 레이어 부터 0~1까지 점점 낮아지는 분포

--------HE-----------
3, 0 -> 특이함. 뒤쪽으로 갈수록 분포가 갈라지는 형태로 생김... 도대체 왜이런 건지?
3, 1 -> 0과 가까운 값에서 극단적인 분포. 뒤의 숫자들은 낮은 수의 값들이 밀집. 
3, 2 -> 역시좀 특이한데.. 직접해보는 게 좋을듯. 첫레이어 pdf -> 단조 증가 함수(지수 증가형), 뒤쪽은 u.r.v.에 가까운 분포.

'''

x = input_data # -> 1000개 data
        
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    if exp[0] == 0:
        # 초깃값을 다양하게 바꿔가며 실험해보자！
        w = np.random.randn(node_num, node_num) * 1
        # 0. N(0, 1)의 분포를 가진 가중치 초기화
    elif exp[0] == 1:
        w = np.random.randn(node_num, node_num) * 0.01
        # 1. N(0, 0.01)의 분포를 가진 가중치 초기화
    elif exp[0] == 2:
        w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
        # 2. Xavier initialization ~ N(n, sqrt(1 /n))
    elif exp[0] == 3:
        w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
        # 3. He initialization ~ N(n, sqrt(2 / n))

    a = np.dot(x, w)


    # 활성화 함수도 바꿔가며 실험해보자！
    if exp[1] == 0:
        z = sigmoid(a)
    elif exp[1] == 1:
        z = ReLU(a)
    elif exp[1] == 2:
        z = tanh(a)

    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
