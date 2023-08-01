data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]

change = True
for i in range(len(data)):
    if not change:
        break
    change = False
    
    for j in range(len(data) - i - 1):
        if data[j] > data[j + 1]:
            data[j], data[j + 1] = data[j + 1], data[j]
            change = True

# 사전에 정렬된 자료에 대한 처리를 좀더 쉽게 함.
# 이경우 복잡도가 O(n)으로 감소함.

print(data)