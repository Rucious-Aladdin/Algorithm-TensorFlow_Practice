data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]

for i in range(len(data)):
    min = i
    for j in range(i + 1, len(data)):
        if data[min] > data[j]:
            min = j
            
    data[i], data[min] = data[min], data[i] #무슨 베릴로그마냥 튜플로 쓰니까 바뀌네 ㄷㄷ..
    
print(data)
