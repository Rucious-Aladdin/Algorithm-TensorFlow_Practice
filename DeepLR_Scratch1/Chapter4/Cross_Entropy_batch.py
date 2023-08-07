import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1: # 차원 배열인 경우의 처리 방식 변경
        t = t.reshape(1, t.size) # 1, x의 크기를 가진 이차원 배열(행렬)로 취급하도록함.
        y = y.reshape(1, y.size)
    
    batch__size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch__size

def cross_entropy_error_num(y, t):#원핫이 아닌경우
    if y.ndim == 1: # 여긴 동일
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch__size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch__size), t])) / batch__size
#np.arange로 ... --> [0, 1, 2, ..., batchsize-1]인 배열 생성
#t는         ... --> [2, 5, 8, ..., pre[batchsize - 1]] 인 (정답 매트릭스 생성)
# np.log(y[[0, 2], y[1, 5] ... )) 와 같이 처리되고 이배열의 값을 모두 합한뒤
# batch_size로 나누어 평균화 해줌.