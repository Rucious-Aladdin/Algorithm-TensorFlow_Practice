data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]


for i in range(len(data)):
# 1회반복 -> 맨끝의 요소(오른쪽)가 최댓값이 됨.
    for j in range(len(data) - (i + 1)):
    #정령된 부분을 제외하고 반복함
        if data[j] > data[j + 1]:
            data[j], data[j + 1] = data[j + 1], data[j]
            
print(data)