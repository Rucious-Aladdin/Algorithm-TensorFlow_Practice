import random

goal = [0b111000000, 0b000111000, 0b000000111, 0b100100100,
        0b010010010, 0b001001001, 0b100010001, 0b001010100]

def check(player):
    for mask in goal:
        if player & mask == mask:
            return True
    return False

def play(p1, p2):
    if check(p2):
        print([bin(p1), bin(p2)])
        print("o나 x가 3개 나열되었습니다.")
        return
    
    board = p1 | p2
    if board == 0b111111111:
        print([bin(p1), bin(p2)])
        print("무승부입니다.")
        return
    
    w = [i for i in range(9) if (board & (1 << i)) == 0]
    print(w)
    r = random.choice(w)
    play(p2, p1 | (1 << r)) #p1 | (1 << r)을 통해서 p1이 수를 둔것임.
    # 다음번에 함수가 호출될때(2번째수)는 p2 | (1 << r)이 인수가 될것이고 이는 p2가 수를 둔것이 됨.
    
play(0 , 0) 