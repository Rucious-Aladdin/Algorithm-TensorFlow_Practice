import random

goal = [0b111000000, 0b000111000, 0b000000111, 0b100100100,
        0b010010010, 0b001001001, 0b100010001, 0b001010100]

def check(player):
    for mask in goal:
        if player & mask == mask:
            return True
    return False

def minmax(p1, p2, turn):
    if check(p2):
        if turn:
            return 1
        else:
            return -1
    
    board = p1 | p2
    if board == 0b111111111:
        return 0
    
    w = [i for i in range(9) if (board & (1 << i)) == 0]
    # 컴퓨터의 입장에서 생각해야함.
    # 1. 레벨 3의 트리를 생각하고, 가장 하위 노드에 경기결과 평가값이 있다고 치자.
    # 2. 나 -> 사람 -> 나 순서 이므로, 처음에 minmax가 리턴될때는 내차례라는 거지.
    # 3. 그러면, 나에게 가장 유리한 선택을 해야하니까, 평가값이 높은 노드를 선택해야 겠네.
    # 4. 나 -> 사람 까지는 해결되었어. 레벨 1의 트리로 끌어 올릴때, 사람이 나에게 가장 불리한 수를 둔다고 생각 해야 겠지?
    # 5. 그렇다면, 평가값의 최솟값을 선택 해야만해.
    # 6. 그다음은 나의 차례야. 이중에 평가값이 가장 높은 수를 내 다음수로 선택하면 되겠네.
    # (이상 전지적 컴퓨터의 시점이었음.)
    if turn:
        # 휴먼 차례일떄는 가장 평가값이 낮은 선택을 해야함.
        x = [minmax(p2, p1 | (1 << i), not turn) for i in w]
        return min(x)
    else:
        # 내 차례 일 때는 평가값이 높은 선택을 함.
        x = [minmax(p2, p1 | (1 << i), not turn) for i in w]
        return max(x)

def printBoard(p1, p2):
    p1_str = format(p1, '09b')
    p2_str = format(p2, '09b')
    p1_str = list(p1_str)
    p2_str = list(p2_str)
    
    empty_board = [["-", "-", "-"], ["-", "-", "-"], ["-", "-", "-"]]
    
    for i in range(9):
        if p1_str[i] == "1":
            empty_board[(8-i) // 3][(8-i) % 3] = "O"
        if p2_str[i] == "1":
            empty_board[(8-i) // 3][(8-i) % 3] = "X"
    for i in range(3):        
        print(empty_board[i])
    
    return

def play(p1, p2, turn):
    
    if check(p2):
        printBoard(p1, p2)
        print("\n너 짐 ㅋㅋ")
        return
    
    board = p1 | p2
    
    if board == 0b111111111:
        printBoard(p1, p2)
        print("잘비볐노 게이야.")
        return

    w = [i for i in range(9) if (board & (1 << i)) == 0]
    
    if not turn:    
        r = [minmax(p2, p1 | (1 << i), True) for i in w]
        #r의 형태: [0, 1, -1, -1] 이런식으로 되어있음. 
        # 리스트의 길이는 현재 둘 수 있는 수만큼임.
        print(r)
        i = [i for i, x in enumerate(r) if x == max(r)]
        j = w[random.choice(i)]
    else:
        print("현재 둘수 있는 위치: ", w)
        printBoard(p1, p2)
        j = int(input("어디에 두시겠습니까?: "))
        
    play(p2, p1 | (1 << j), not turn)

#선후공 선택
print("선후공을 선택하세요: ")
a = int(input("1. 선공 2. 후공: "))

if (a == 1):
    print("True")
    play(0, 0, True)
else:
    play(0, 0, False)

#    