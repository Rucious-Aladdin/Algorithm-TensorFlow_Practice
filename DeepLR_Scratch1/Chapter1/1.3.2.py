#print types of data
print(type(10))
print(type(2.718))
print(type("hello"))

#print variable
x = 10
print(x)
x = 100
print(x)
y = 3.14
print(3.14)

print(x * y)
print(type(x * y))

# list data type

a = [1, 2, 3, 4, 5]
print(a)

print(len(a))
print(a[0])
print(a[4])
a[4] = 99

print(a)

print(a[0:2])
print(a[1:])
print(a[:3])
print(a[:-1])
print(a[:-2])

#Dictionary data type

me = {"height" : 180}
print(me["height"])

me["weight"] = 70
print(me)

hungry = True
sleepy = False

print(type(hungry))
print(not hungry)
print(hungry and sleepy)
print(hungry or sleepy)

#if 문

hungry = False
if hungry:
    print("I'm happy")
else:
    print("I'm not hungry")
    print("I'm sleepy")

#for 문

for i in [1, 2, 3]:
    print(i)


def hello():
    print("Hello World!")
    
hello()

def hello_obj(object):
    print("Hello " + object + "!")
    
hello_obj("cat")