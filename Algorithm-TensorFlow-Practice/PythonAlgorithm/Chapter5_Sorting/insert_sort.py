data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]

for i in range(1, len(data)):
    temp = data[i] #현재 조사중인 요소를 일시적으로 저장함.
    j = i - 1 # 직전의 위치를 j에 저장함.
    while (j >= 0) and (data[j] > temp):
    #이언어는 파이썬이므로.. j>=0이 있어야 한다는 것.
    # temp가 앞선 리스트의 자료보다 크면 한칸씩 오른쪽으로 옮기고,
    # 처음으로 temp가 큰숫자가 등장하면 while문이 실행이 되지않고, j+1자리에 temp가 들어가게 됨.
        data[j + 1] = data[j]
        j -= 1
    data[j + 1] = temp

print(data)

## 이진검색으로 index를 찾는것(앞부분은 정렬되어 있기때문에) -> 일반적으로 안함. 결국 리스트의 자료를 옮기는데 시간이 걸리기 때문
## 연결리스트를 이용해 삽입시간 단축? -> 리스트의 index를 찾는데 이진검색 활용이 불가능하다는 점이 문제임.