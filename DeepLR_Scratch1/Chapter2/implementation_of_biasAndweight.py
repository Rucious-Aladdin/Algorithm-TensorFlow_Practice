import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7 #bias가 b로 치환됨.
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp<= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp<= 0:
        return 0
    else:
        return 1

test = [[0, 0], [0, 1], [1, 0], [1, 1]]

for i in test:
    print("AND: " + f"{(i[0], i[1])}" + " " + str(AND(i[0], i[1])))
    
for i in test:
    print("NAND: " + f"{(i[0], i[1])}" + " " + str(NAND(i[0], i[1])))
    
for i in test:
    print("OR: " + f"{(i[0], i[1])}" + " " + str(OR(i[0], i[1])))

#EZ