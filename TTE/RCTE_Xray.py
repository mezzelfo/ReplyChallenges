from more_itertools import split_at
import sys
sys.setrecursionlimit(15000)
def func(s):
    base = min(s)
    s = [k - base for k in s]
    chunks = list(split_at(s, lambda x: x == 0))
    chunks = [func(k) for k in chunks if len(k)>0]
    return base+sum(chunks)


with open('input.txt','r') as f:
    testCaseCount = int(f.readline())
    for testCase in range(testCaseCount):
        cmCount = int(f.readline())
        cmDose = [int(k) for k in f.readline().split(' ')]
        print('Case #'+str(testCase+1)+':',func(cmDose))
