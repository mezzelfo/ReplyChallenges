import numpy as np
from os import listdir
import matplotlib.pyplot as plt

class Replayer:
    def __init__(self):
        self.isManager = False
        self.company = ""
        self.bonus = -1
        self.skills = {}
        self.coord = [-1, -1]
    def __repr__(self):
        if self.isManager:
            s="MNG "
        else:
            s="DEV "
        s+=self.company+" "
        s+=str(self.coord)+" "
        s+=str(len(self.skills))
        return s


def read_developer(line):
    r = Replayer()
    data = line.split()
    r.company = data[0]
    r.bonus = int(data[1])
    r.skills = set(data[3:])
    assert len(r.skills) == int(data[2])
    return r


def read_manager(line):
    r = Replayer()
    data = line.split()
    r.company = data[0]
    r.bonus = int(data[1])
    r.skills = set(data[3:])
    assert len(r.skills) == 0
    r.isManager = True
    return r


def readinput(filename):
    with open(filename, 'r') as file:
        filecontent = file.readlines()

    W, H = [int(x) for x in filecontent[0].strip().split()]
    matrix = np.array([list(s.strip()) for s in filecontent[1:H+1]])
    assert matrix.shape == (H, W)
    #matrix = matrix.tolist()

    D = int(filecontent[H+1].strip())
    developers = []
    for line in filecontent[H+2:H+2+D]:
        developers.append(read_developer(line.strip()))
    assert len(developers) == D

    M = int(filecontent[H+D+2].strip())
    managers = []
    for line in filecontent[H+3+D:]:
        managers.append(read_manager(line.strip()))
    assert len(managers) == M

    return W, H, D, M, matrix, developers, managers

def readoutput(filename, D, M):
    with open(filename+'.out', 'r') as file:
        filecontent = file.readlines()
    dev_pos = []
    for line in filecontent[:D]:
        if 'X' not in line:
            dev_pos.append((int(x) for x in line.strip().split()))
        else:
            dev_pos.append((-1,-1))
    man_pos = []
    for line in filecontent[D:]:
        if 'X' not in line:
            man_pos.append((int(x) for x in line.strip().split()))
        else:
            man_pos.append((-1,-1))
    return dev_pos, man_pos

for filename in sorted(listdir()):
    if '.txt' in filename and 'a_' not in filename and '.out' not in filename:
    #if 'b_dream.txt' == filename:
        print('Opening',filename)
        W, H, D, M, matrix, developers, managers = readinput(filename)
        dev_pos, man_pos = readoutput(filename,D,M)
        assert len(dev_pos) == D and len(man_pos) == M
        for (x,y) in dev_pos:
            if (x,y) == (-1,-1):
                continue
            assert matrix[y,x] == '_'
        for (x,y) in man_pos:
            if (x,y) == (-1,-1):
                continue
            assert matrix[y,x] == 'M'
        s1 = [p for p in dev_pos if p!=(-1,-1)]
        assert len(s1) == len(set(s1))

        s2 = [p for p in man_pos if p!=(-1,-1)]
        assert len(s2) == len(set(s2))

        assert len(set(s1).intersection(set(s2))) == 0