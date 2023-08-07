class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy

class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = 1 * dout
        dy = 1 * dout
        
        return dx, dy
        

if __name__ == "__main__":
    apple = 100
    apple_num = 2
    tax = 1.1
    
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()
    
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)
    
    print(price)
    #스택처럼 나중에 들어가는 층을 먼저 호출해서 값을 구해야함.
    dprice = 1 # 가장 상위 계층의 미분값을 넣어줌.
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    
    print(dapple, dapple_num, dtax)
    
    ## 귤 사과 예제 구현
    
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1
    
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)
    
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    
    print(price)
    print(dapple_num, dapple, dorange, dorange_num, dtax)