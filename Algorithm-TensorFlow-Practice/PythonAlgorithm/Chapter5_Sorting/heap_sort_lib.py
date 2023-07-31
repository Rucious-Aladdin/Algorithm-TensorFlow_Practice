import heapq

def heap_sort(array):
    h = array.copy()
    heapq.heapify(h)  # array를 힙으로 만들어 버림.
    return [heapq.heappop(h) for _ in range(len(array))] #데이터를 꺼내면서 동시에 정렬함.

data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]

print(heap_sort(data))