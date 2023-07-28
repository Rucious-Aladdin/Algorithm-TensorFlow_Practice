tree = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
        [], [15, 16], [], [], [], [], [], [], [], []]

def search(pos):
    # pos = 초기 탐색위치
    print(pos, end = " ")
    for i in tree[pos]:
        search(i)

search(0)