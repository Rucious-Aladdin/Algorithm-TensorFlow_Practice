from numerical_grad import numerical_gradient
import numpy as np

def gradient_descent(f, init_x, lr = 0.01, step_num=100):
    x = init_x
    
    for i in range(100):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        print("{:.6f} {:.6f}".format(x[0].item(), x[1].item()))
        
    return x

def func(x):
    return x[0] ** 2 + x[1] ** 2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(func, init_x=init_x, lr = 0.1, step_num=100))

init_x = np.array([-3.0, 4.0])
a = gradient_descent(func, init_x=init_x, lr = 10.0, step_num=100) #발사 해버림
print(a)

init_x = np.array([-3.0, 4.0])
print(gradient_descent(func, init_x=init_x, lr = 1e-2, step_num=100)) # 값이 거의 변하지 않아 버림.
