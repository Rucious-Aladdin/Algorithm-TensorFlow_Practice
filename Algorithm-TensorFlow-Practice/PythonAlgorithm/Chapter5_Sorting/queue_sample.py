import queue

q = queue.Queue()

#Queue의 method -> put, get 두개만 외우면 됨!

q.put(3)
q.put(5)
q.put(2)

temp = q.get()
print(temp)
temp = q.get()
print(temp)

q.put(4)

temp = q.get()
print(temp)