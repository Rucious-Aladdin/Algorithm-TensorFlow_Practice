#힙의 구조

# 부모노드의 수가 그 부모의 모든 자식노드의 수보다 작아야 함(반대도 가능하긴 함.)

"""
1. 힙에서 요소를 꺼내기

(1) 최솟 값의 경우 -> 루트노드에 최솟값이 항상 존재하므로 그걸 빼내고,
(2) 마지막 요소 값을 루트노드에 넣고 힙을 재정렬 함.

2. 힙에 요소를 넣기.
(1) 마지막 요소에 값을 할당한 뒤에
(2) 힙을 재정렬 함.

- 힙의 재정렬

* 부모노드와 자식노드의 값을 서로 비교해서, 한개 레벨 씩 위 아래로 값이 이동함.
복잡도는 O(log(n))에 해당함.

"""

data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]

for i in range(len(data)):
    j = i
    # (j - 1) // 2 는 현재 위치한 node(j)의 부모노드의 index에 해당함.
    while (j > 0) and (data[(j - 1) // 2] < data[j]):
        # j
        data[(j - 1) // 2], data[j] = data[j], data[(j - 1) // 2]
        j = (j - 1) // 2
        
print(data)
for i in range(len(data), 0, -1):
    # list의 맨뒤에서 부터 시작함.
    data[i - 1], data[0] = data[0], data[i - 1]
    j = 0
    
    while((2 * j + 1 < i - 1) and (data[j] < data[2 * j + 1]) 
          or (2 * j + 2 < i - 1) and (data[j] < data[2 * j + 2])):
        # 최댓값을 가장 상위 루트 노드로 올리는 알고리즘을 가지고 있음.
        if (2 * j + 2 == i - 1) or (data[2 * j + 1] > data[2 * j + 2]):
            data[j], data[2 * j + 1] = data[2 * j + 1], data[j]
            j = 2 * j + 1            
        else:
            data[j], data[2 * j + 2] = data[2 * j + 2], data[j]
            j = 2 * j + 2 
print(data)