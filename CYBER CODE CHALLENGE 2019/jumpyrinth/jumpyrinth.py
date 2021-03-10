import numpy as np

data = []
up = np.asarray([-1,0])
down = np.asarray([1,0])
left = np.asarray([0,-1])
right = np.asarray([0,1])

def leggiNumero(initial,direction):
    pos = initial+direction
    digits = []
    while data[pos[0],pos[1]].isdigit():
        digits.append(data[pos[0],pos[1]])
        pos += direction

    if np.all(direction == left) or np.all(direction == up):
        digits.reverse()

    n = int(''.join(str(i) for i in digits))
    return n

with open('2c464e58-9121-11e9-aec5-34415dec71f2.txt') as file:
    for line in file:
        data.append(list(line))
    data[-1].append('\n')
    data = np.asmatrix(data)

xstart, ystart = np.where(data == '$')

for startPos in zip(xstart, ystart):
    flag = []
    stack = []
    pos = np.asarray(startPos)
    while(data[pos[0],pos[1]] != '@'):
        cmd = data[pos[0],pos[1]]
        #print(pos,cmd)
        if cmd == '$':
            print('Ho iniziato')
            pos += down
        elif cmd == '#':
            print('null')
            break
        elif cmd == '(' or cmd == '<':
            if cmd == '(':
                if len(stack)==0:
                    print('empty stack pop')
                    break
                flag.insert(0,stack.pop())
            n = leggiNumero(pos, right)
            pos += n*left
        elif cmd == ')' or cmd == '>':
            if cmd == ')':
                if len(stack)==0:
                    print('empty stack pop')
                    break
                flag.insert(-1,stack.pop())
            n = leggiNumero(pos, left)
            pos += n*right
        elif cmd == '-' or cmd == '^':
            if cmd == '-':
                flag = flag[1:]
            n = leggiNumero(pos, down)
            pos += n*up
        elif cmd == '+' or cmd == 'v':
            if cmd == '+':
                flag = flag[:-1]
            n = leggiNumero(pos, up)
            pos += n*down
        elif cmd == '%':
            flag.reverse()
            pos += down
        elif cmd == '[':
            pos += right
            stack.append(data[pos[0],pos[1]])
            pos += right
        elif cmd == ']':
            pos += left
            stack.append(data[pos[0],pos[1]])
            pos += left
        elif cmd == '*':
            pos += up
            stack.append(data[pos[0],pos[1]])
            pos += up
        elif cmd == '.':
            pos += down
            stack.append(data[pos[0],pos[1]])
            pos += down
        else:
            print('error')
            break

        if np.all(pos == [57,954]):
            break

        if (pos[0] >= 1024 or pos[1] >= 1025):
            print('out of bound')
            break
    print(stack)
    print(flag)
    #input()


        