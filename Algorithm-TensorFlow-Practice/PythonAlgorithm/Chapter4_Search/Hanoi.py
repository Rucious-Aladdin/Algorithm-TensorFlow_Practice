def hanoi(n, src, dist, via):
    #n = 원판수, src = 현재위치, dist = 이동 위치, via = 경유장소
    if n > 1:
        hanoi(n - 1, src, via, dist)
        print(src + " -> " + dist)
        hanoi(n - 1, via, dist, src)
    else:
        print(src + " -> " + dist)
    
n = int(input())
hanoi(n, "a", "b", "c")