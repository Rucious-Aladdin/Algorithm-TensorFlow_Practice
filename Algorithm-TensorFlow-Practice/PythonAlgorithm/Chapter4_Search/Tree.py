tree = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
        [], [], [], [], [], [], [], []]

data = [0]

while len(data) > 0:
    pos = data.pop(0)
    print(pos, end = " ")
    print(data) # current searching nodes
    # 특정 node에 도착 -> node의 child node를 앞으로 돌게될 node의 뒷부분으로 추가함.
    for i in tree[pos]:
        data.append(i)
        
print(data)