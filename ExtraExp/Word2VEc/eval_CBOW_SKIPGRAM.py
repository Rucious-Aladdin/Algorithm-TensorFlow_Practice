import matplotlib.pyplot as plt
import numpy as np

sem =[0.033402922755741124, 0.021920668058455117, 0.028183716075156576, 0.032359081419624215, 0.028183716075156576, 0.024008350730688934, 0.018789144050104383] 
sem_top5 = [0.04697286012526096, 0.04488517745302714, 0.05010438413361169, 0.05323590814196242, 0.04175365344467641, 0.04801670146137787, 0.054279749478079335]
syn = [0.020876826722338204, 0.027139874739039668, 0.04384133611691023, 0.05219206680584551, 0.049060542797494784, 0.04592901878914405, 0.05219206680584551] 
syn_top5 = [0.0720250521920668, 0.11795407098121086, 0.174321503131524, 0.1847599164926931, 0.21294363256784968, 0.17536534446764093, 0.20146137787056367]


SKIP_time_f_window = [5423.171353593003, 3946.8330044350587, 3466.88645546115, 3048.387870428851, 2633.8078976429533, 2721.618077710038, 2805.0643430398777]
SKIP_time_f_hidden = [756.6754868228454, 1347.1638888730668, 2044.2699193998706, 2749.2740077320486, 2969.320798396133]
CBOW_time_f_window = [574.0027752178721, 382.5598237449303, 403.3865006980486, 405.7648449279368, 394.0074226630386, 454.121841032058, 457.23553821304813]
CBOW_time_f_hidden = [277.6534899568651, 313.63740395288914, 351.05838014394976, 390.89739393722266, 432.06839760695584]

s1 = sum(SKIP_time_f_hidden)
s2 = sum(SKIP_time_f_window)
s3 = sum(CBOW_time_f_hidden)
s4 = sum(CBOW_time_f_window)

print((s1 + s2) / (s3 + s4)) 


sem = np.array(sem)
sem_top5 = np.array(sem_top5)
syn = np.array(syn)
syn_top5 = np.array(syn_top5)


x_data = len(sem)

x_1 = [2, 4, 6, 8, 10]
x_2 = [25, 50, 100, 200, 300, 400, 500]

plt.subplot(121)
plt.plot(x_1, CBOW_time_f_hidden, label="CBOW", color="green")
plt.plot(x_1, SKIP_time_f_hidden, label="SKIP-GRAM", color="red")
plt.legend(loc="upper left")
plt.xlabel("window size")
plt.ylabel("time (s)")
plt.grid(True)
plt.title("Training Time (fixed window size)")

plt.subplot(122)
plt.plot(x_2, CBOW_time_f_window, label="CBOW", color="green")
plt.plot(x_2, SKIP_time_f_window, label="SKIP-GRAM", color="red")
plt.legend(loc="upper right")
plt.xlabel("window size")
plt.ylabel("time (s)")
plt.grid(True)
plt.title("Training Time (fixed hidden size)")

plt.show()



#word analogy task
"""
plt.subplot(121)
plt.plot(x, sem, label="sem. acc", color="green")
plt.plot(x, sem_top5, label="sem. top5-acc", color="limegreen")
plt.plot(x, syn, label="syn. acc", color="red")
plt.plot(x, syn_top5, label="syn. top5-acc", color="pink")
plt.legend(loc="upper right")
plt.xlabel("window size")
plt.ylabel("accuracy")
plt.grid(True)
plt.ylim(0, 0.35)
plt.title("semantic / syntactic analysis")

plt.subplot(122)
plt.plot(x, (syn + sem) / 2, label="overall acc", color="blue")
plt.plot(x, (syn_top5 + sem_top5) / 2, label="overall top5-acc", color="skyblue")
plt.legend(loc="upper right")
plt.xlabel("window size")
plt.ylabel("accuracy")
plt.title("overall analysis")
plt.grid(True)
plt.ylim(0, 0.35)

plt.tight_layout()
plt.show()
"""
