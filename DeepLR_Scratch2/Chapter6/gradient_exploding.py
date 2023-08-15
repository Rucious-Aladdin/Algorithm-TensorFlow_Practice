import numpy as np
import matplotlib.pyplot as plt

N = 2
H = 3
T = 20

dh = np.ones((N, H))

Wh = np.random.randn(H, H)

norm_list = []

for i in range(10):
    norm_list = []
    for t in range(T):
        dh = np.matmul(dh, Wh.T)
        norm = np.sqrt(np.sum(dh ** 2)) / N
        norm_list.append(norm)
        
    plt.plot(range(T), norm_list)
    plt.show()