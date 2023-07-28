city_dict = {1:3416918, 2:1525849, 3:1147037, 4:1044579, 5:828947, 6:654963, 7:565392, 8:506494,
             9:2925967, 10:1496172, 11:1068641, 12:942649, 13:818760, 14:652845, 15:542713, 16:489202,
             17:2453041, 18:1193894, 19:1061440, 20:840047, 21:702545, 22:650599, 22:521642}
global_ans = []
global_val = 5000000

def search(city_dict, answer, value=5000000, level = 0):
    global global_ans
    global global_val
    if level == 3:
        value = abs(city_dict[answer[0]] + city_dict[answer[1]] + city_dict[answer[2]] - 5000000)
        if abs(city_dict[answer[0]] + city_dict[answer[1]] + city_dict[answer[2]] - 5000000) < global_val:
            global_ans = answer
            global_val = value
        return
    else:
        level += 1
        for i in range(len(city_dict)):
            if (i + 1) not in answer:    
                search(city_dict, answer + [i + 1], value, level)
                
search(city_dict, [])
print(global_val, global_ans)
print(city_dict[1] + city_dict[19] + city_dict[22])