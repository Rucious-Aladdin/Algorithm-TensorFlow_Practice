import math, time

def get_prime(n):
    if n <= 1:
        return []
    prime = [2]
    limit = int(math.sqrt(n))
    
    data = [i + 1 for i in range(2, n, 2)]
    # range(시작할 수, 끝수, stepsize)
    
    while limit >= data[0]:
        prime.append(data[0])
        data = [j for j in data if j % data[0] != 0]
        # y for y in list1 (if 조건)
        # if 조건이하를 만족하는 list1의 object y를 list에 for문이 끝날때 까지 append함
    return prime + data

print(get_prime(200))

start = time.time()
print(get_prime(100000))
end = time.time()

print(f"{end - start: .5f} sec")