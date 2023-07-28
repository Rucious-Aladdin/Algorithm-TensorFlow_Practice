def convert(n, base = 2):
    result = ""
    while n > 0:
        result = str(n % base) + result
        n //= base
    
    return result

n = int(input("input num: "))

print(convert(n))
print(convert(n, 8))
print(convert(n, 3))