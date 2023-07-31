data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]

for i in range(1, len(data)):
    temp = data[i] #현재 조사중인 요소를 일시적으로 저장함.
    j = i - 1 # 직전의 위치를 j에 저장함.
    while (j >= 0) and (data[j] > temp):
        data[j + 1] = data[j]
        j -= 1
    data[j + 1] = temp

print(data)