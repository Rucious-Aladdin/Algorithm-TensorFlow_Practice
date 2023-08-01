data = [6, 15, 4, 2, 8, 5, 11, 9, 7, 13]

def merge_sort(data):
    if len(data) <= 1:
    # 재귀함수의 탈출 조건.
    # 리스트의 분할이 완료되면 탈출.
        return data
    
    mid = len(data) // 2
    
    left = merge_sort(data[:mid])
    right = merge_sort(data[mid:])
    
    return merge(left, right)

def merge(left, right):
    print(left, right)
    result = []
    i, j = 0, 0
    
    while (i < len(left)) and (j < len(right)):
        
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
        
    #한쪽 리스트가 고갈 되었다는건...
    #고갈된 리스트의 값이 고갈되지 않은 모든 리스트의 값보다 작다는 것을 의미.
    #왜? 고갈되지 않은 리스트의 첫번째 요소 뒤쪽은 오름차순으로 정렬되어 있기 때문임.
    
    #따라서.. 한쪽리스트가 모두 고갈되면 고갈되지 않은 리스트의 숫자를 result에 전부 extend시킴.
    if i < len(left):
    #list.append(x)는 리스트 끝에 x 1개를 그대로 넣습니다.
    #list.extend(iterable)는 리스트 끝에 가장 바깥쪽 iterable의 모든 항목을 넣습니다.
    #append와 extend의 차이점에 유의해 코딩을 진행한다.
    #list_type + [] 와 매우 유사한 연산으로 보임.
        result.extend(left[i:])
    if j < len(right):
        result.extend(right[j:])
    print(result)
    return result

print(merge_sort(data))

"""
    연산의 차이점에 대해서 명확히 알아 둘것.
a = ["a"]
b = "bc"

c = a + [b]
print(c)
a.append(b)
print(a)
a.extend(b)
print(a)

"""
