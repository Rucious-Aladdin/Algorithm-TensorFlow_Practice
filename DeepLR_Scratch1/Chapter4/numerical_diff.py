def numerical_diff(f, x):
    h = 1e-4
    # 1e-50 -> rounding error발생 ... float의 precision에 한계가 있기 때문
    return (f(x + h) - f(x)) / h


def numerical_diff_central(f, x):
    h = 1e-4
    # 1e-50 -> rounding error발생 ... float의 precision에 한계가 있기 때문
    return (f(x + h) - f(x-h)) /(2*h)
