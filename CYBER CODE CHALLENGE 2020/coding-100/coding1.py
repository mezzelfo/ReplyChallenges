with open('coding-100/true.txt') as f:
    truebook = f.read()
with open('coding-100/The Time Machine by H. G. Wells.txt') as f:
    falsebook = f.read()

for i,(cT,cF) in enumerate(zip(truebook,falsebook)):
    if cT != cF:
        print(cF,end='')
        lasterror = i