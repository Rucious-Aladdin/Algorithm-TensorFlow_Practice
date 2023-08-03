import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    
    return exp_a / sum_exp_a


if __name__ == "__main__":
    a = np.array([1010, 1000, 990])
    print(np.exp(a) / np.sum(np.exp(a)))
    
    c = np.max(a)
    print(a - c)
    
    print(np.exp(a) / np.sum(np.exp(a - c)))
    print(softmax(a))
    

