# NON-LINEAR영역에 대한 표현을 가능케 하기위한 방법을 말하고자 하는 것임.
from implementation_of_biasAndweight import AND, NAND, OR

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    
    return y

test = [[0, 0], [0, 1], [1, 0], [1, 1]]
