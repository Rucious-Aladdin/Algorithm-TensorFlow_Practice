count = 0

def search(max, current, answer):
    global count
    if current == max:
        count += 1
        print(answer)
        return
    else:
        for i in range(max - current):
            if current + (i + 1) <= max:
                search(max, current + i + 1, answer + [1 + current + i])
#chatGPT형님 한수 배웠씁ㄴ다.
search(10, 1, [1])
print(count)