import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))
print(np.dot(B ,A))
#행렬계산은 교환법칙이 성립안함.

C = np.array([[1, 2], [3, 4]])
print(C.shape)
A = np.array([[1, 2, 3], [4, 5, 6]])

#print(np.dot(A.T, C) 얘는 안됨.
print(np.dot(A.T, C)) #얘는 됨

#벡터와 행렬의 연산.
A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)

B = np.array([7, 8])
print(B.shape)
#dot연산으로 처리해주면 됨.
print(np.dot(A ,B))

X = np.array([[1, 2, 3]])
Y = np.array([[2, 4, 6]])

print(np.dot(X.T, Y) ) #covariance matrix, just Test!