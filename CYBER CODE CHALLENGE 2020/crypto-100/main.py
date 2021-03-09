with open('/home/luca/Documents/ReplyCyber2020/crypto-100/smorfia_tab.txt') as f:
    todecrypt = [l.strip().split(',') for l in f.readlines()]

smorfia = {
    'caffe':42,
    'giardino':51,
    'peli':59,
    'figliolanza':9,
    'ragazzo':15,
    'caciocavalli':88,
    'testa':34,
    'bambina':2,
    'guardie':24,
    'capitone':32,
    'piccolaanna':26,
    'coltello':41,
    'uccello':35,
    'topi':11,
    'maiale':4,
    'mortoammazzato':62,
    'pitale':27,
    'palletenente':30,
    'donnabalcone':43,
    'noia':40,
    'santantonio':13,
    'gatta':3,
    'mano':5
}

def get_dict_key(value):
    for key, val in smorfia.items():
        if val == value:
            return key
    print(value)

key = []
for l in todecrypt:
    if len(l) == 1:
        key.append(smorfia[l[0]])
    else:
        key.append(smorfia[l[0]]*10+smorfia[l[1]])
        #key.append(smorfia[l[0]]+smorfia[l[1]])

from collections import Counter
from string import ascii_lowercase
import matplotlib.pyplot as plt
print(min(key),max(key))
print(key)
#print(Counter([k%27 for k in key]))
#print(''.join([chr(64+k%27) for k in key]))
#print([chr(64+(108-k)%27) for k in key])
#for i in range(0,28):
#    print(''.join([chr(65+(108-k+i)%26) for k in key]))
print(Counter(key))
plt.plot(sorted(set(key)))
plt.show()
#cnt = {list(set(key))[i]:ascii_lowercase[i] for i in range(len(set(key)))}
#print(''.join([cnt[k] for k in key]))
#print([hex(i) for i in key])
#for delta in range(-256,256):
#print(''.join([chr((i-19)%256) for i in key]))
#print(''.join([str(hex(i))[2:] for i in key]))
#message = '727f00340e075a6b3a69146f2d3e3a67403c343e101d052b1a58623d3c1a0e53087c00245b6e00771d1f1005316e08693e24000714'
#message = [int('0x'+message[i:i+2],16) for i in range(0,len(message),2)]
#for delta in range(0,256):
#    print(''.join([chr((i+delta)%256) for i in message]))
#print(message)
# print(len(key))
# print(len(message))
# print(len(set(key)))
# c = list(Counter(key))
# c2 = {c[i]:ascii_lowercase[i] for i in range(len(c))}
# print(''.join([c2[k] for k in key]))