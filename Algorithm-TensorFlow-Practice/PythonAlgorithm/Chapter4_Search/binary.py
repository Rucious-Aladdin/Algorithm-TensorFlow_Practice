def binary_search(data, value):
    # list type data, dim = 1
    # value = target value
    
    left = 0 # index of statting position
    right = len(data) - 1 # index of end position
    count = 0
    while left <= right:
        mid = (left + right) // 2
        count += 1
        if data[mid] == value:
            return mid, count # return index of target value
        elif data[mid] < value:
            left = mid + 1
        else:
            right = mid - 1
        
    return -1, count

data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(binary_search(data, 50))
    
    