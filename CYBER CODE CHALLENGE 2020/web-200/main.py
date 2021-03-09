def f(c):
    return chr(33+(ord(c)-33+47)%94)
#for i in range(33,126+1):
#    print(chr(i),f(i))

#import requests
#r = requests.post('http://gamebox1.reply.it/0b7d3eb5b7973d27ec3adaffd887d0e2/',data={'cipher':'4:2@'})
#print(r.text)
print([f(c) for c in '{FLG:'])