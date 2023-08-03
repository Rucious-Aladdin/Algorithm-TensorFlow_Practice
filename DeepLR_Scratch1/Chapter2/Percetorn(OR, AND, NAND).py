def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    
    if tmp <= theta:
        return 0
    else:
        return 1

def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    tmp = x1 * w1 + x2 * w2
    
    if tmp <= theta:
        return 0
    else:
        return 1

def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.1
    tmp = x1 * w1 + x2 * w2
    
    if tmp <= theta:
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