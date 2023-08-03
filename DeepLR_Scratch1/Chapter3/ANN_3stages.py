import numpy as np
from Sigmoid_Func import sigmoid

#출력층의 활성화 함수를 정의한 것.
def identity_function(x):
    return x

if __name__ == "__main__":
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    #2차원 배열로 만들어야됨. 실수좀 그만해라!
    B1 = np.array([0.1, 0.2, 0.3])

    print(X.shape)
    print(W1.shape)
    print(B1.shape)

    A1 = np.dot(X, W1) + B1
    print(A1)
    Z1 = sigmoid(A1)

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])

    print(Z1.shape)
    print(W2.shape)
    print(B2.shape)

    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    print(Z2)
    Y = identity_function(Z2)