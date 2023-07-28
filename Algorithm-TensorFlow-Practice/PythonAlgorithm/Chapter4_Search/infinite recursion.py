def re(count):
    if count == 100:
        return
    print(count)
    re(count + 1)


re(0)