#일반적인 재귀로 이루어진 fibonacci

memo = {1:1, 2:1}

def fibonacci(n):
    if n in memo:
        return memo[n]
    memo[n] = fibonacci(n-2) + fibonacci(n-1)
    return memo[n]

print(fibonacci(1001) / fibonacci(1000))
###