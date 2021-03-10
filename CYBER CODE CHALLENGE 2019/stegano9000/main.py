from math import floor

def f1(x):
    return floor(0.27*x+269)

def f2(x):
    return floor(0.44*x+163)

eq = []
for x in range(1000):
    if f1(x) == f2(x):
        eq.append(x)
print(eq)