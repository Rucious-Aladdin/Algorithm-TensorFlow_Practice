def numerical_diff(f, x):
    h = 1e-50
    return (f(x + h) - f(x)) / h

x = 0.333333
print(x == 0.333333)