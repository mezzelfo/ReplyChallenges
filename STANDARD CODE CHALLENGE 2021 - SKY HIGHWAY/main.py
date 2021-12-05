from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import cv2.cv2 as cv2
from collections import Counter
import ctypes
import scipy.optimize

def linetoint(line):
    return [int(n) for n in line.strip().split(' ')]

def getKernels(radius):
    v = np.arange(radius)
    v = np.append(v,np.arange(radius,-1,-1))
    manhattan = 2*radius-np.add.outer(v,v)
    inside = 1*(manhattan <= radius)
    manhattan = manhattan*inside
    return manhattan, inside

total = 0
for filename in sorted(listdir()):
    if '.in' in filename:
    #if 'data_scenarios_c_metropolis.in' in filename:
    #if 'data_scenarios_a_example.in' in filename:
        with open(filename) as f:
            print(filename)
            W,H = linetoint(f.readline())
            N,M,R = linetoint(f.readline())
            buildings = np.asarray([linetoint(f.readline()) for _ in range(N)])
            antennas = np.asarray([linetoint(f.readline()) for _ in range(M)])
            print(W,H,N,M,R, np.max(antennas[:,0]))
            #print(antennas)
        # buildings = buildings.astype(np.int32).flatten()
        # antennas = antennas.astype(np.int32).flatten()
        # c_int_p = ctypes.POINTER(ctypes.c_int32)
        # buildings_p = buildings.ctypes.data_as(c_int_p)
        # antennas_p = antennas.ctypes.data_as(c_int_p)
        # antennasPos = np.random.randint(0,high=15,size=(M,2),dtype = np.int32).flatten()
        # antennasPos[0] = 10
        # antennasPos[1] = 7
        # antennasPos[2] = 12
        # antennasPos[3] = 2
        # antennasPos[4] = 2
        # antennasPos[5] = 4
        # antennasPos[6] = 0
        # antennasPos[7] = 7
        # antennasPos_p = antennasPos.ctypes.data_as(c_int_p)

        # scorelib = ctypes.CDLL('/home/luca/Documents/ReplyChallenges/STANDARD CODE CHALLENGE 2021 - SKY HIGHWAY/scorelib.so')
        # scorefun = scorelib.score
        # scorefun.argtypes = [c_int_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,c_int_p,c_int_p]
        # res = scorefun(N,M,R,buildings_p,antennas_p,antennasPos_p)
        # print(res)


        # dimensions = np.max(buildings[:,[0,1]], axis = 0)
        # connectionspeed_board = np.zeros(dimensions+1)
        # connectionspeed_board[buildings[:,0],buildings[:,1]] = buildings[:,3]
        # latency_board = np.zeros(dimensions+1)
        # latency_board[buildings[:,0],buildings[:,1]] = buildings[:,2]
        # kernels = {r: getKernels(r) for r in np.unique(antennas[:,0])}
        # usedpos = {}
        # for ant in antennas: #TODO: choose ordering
        #     manhattan, inside = kernels[ant[0]]
        #     diffused_latency_board = cv2.filter2D(latency_board,-1,manhattan)
        #     diffused_connectionspeed_board = cv2.filter2D(connectionspeed_board,-1,inside)
        #     scoremap = diffused_connectionspeed_board*ant[1]-diffused_latency_board
        #     bestpos = np.unravel_index(np.argmax(scoremap,axis=None),scoremap.shape)
        #     if scoremap[bestpos] > 0:
        #         if bestpos in usedpos:
        #             if usedpos[bestpos][0] < scoremap[bestpos]:
        #                 usedpos[bestpos] = (scoremap[bestpos],tuple(ant))
        #         else:
        #             usedpos[bestpos] = tuple(ant)
        #     #print(np.max(scoremap))
        #     #cv2.imshow('a',scoremap)
        #     #cv2.waitKey()

        
        # if np.all(antennas[:,0] == 0):
        #     sortedbuildingsidx = np.argsort(buildings[:,-1])[::-1]
        #     sortedantennaidx = np.argsort(antennas[:,1])[::-1]
        #     #sa = antennas[sortedantennaidx[0:M],:]
        #     sb = buildings[sortedbuildingsidx[0:M],:]
        #     print(M)
        #     for i in range(M):
        #         idx = np.where(sortedantennaidx == i)[0][0]
        #         print(i,sb[idx,0],sb[idx,1])
        #     # score = np.dot(sa[:,-1],sb[:,-1])+R*(M>=N)
        #     # print(score)
        #     # total += score
        #     # print(total)

        # plt.subplot(2,2,1)
        # plt.title('Antennas Range')
        # plt.hist(antennas[:,0])
        # plt.subplot(2,2,2)
        # plt.title('Antennas Connection Speed')
        # plt.hist(antennas[:,1])
        # plt.subplot(2,2,3)
        # plt.title('Buildings Latency')
        # plt.hist(buildings[:,2])
        # plt.subplot(2,2,4)
        # plt.title('Buildings Connection Speed')
        # plt.hist(buildings[:,3])
        # plt.suptitle(filename)
        
        # maxx = np.max(buildings[:,0])
        # maxy = np.max(buildings[:,1])
        # board = np.zeros((maxx+1,maxy+1), dtype=np.uint8)
        # for b in buildings:
        #     board[b[0],b[1]] = int(255*b[2]/np.max(buildings[:,2]))
        # cv2.imshow(filename,board)
        # cv2.waitKey()
            
        # maxx = np.max(buildings[:,0])
        # maxy = np.max(buildings[:,1])
        # print('\t',W,H,N,M,R)
        # print('\t','Space',maxx,maxy)
        
        # trueareas = (2*antennas[:,0]+1)**2
        # print('\t','Total coverable area',np.sum(trueareas))
        # board = np.zeros((maxx+1,maxy+1), dtype=np.uint8)
        # board[buildings[:,0],buildings[:,1]] = 255
        # blurred_board = cv2.blur(board, (1,1))
        # blurred_board[blurred_board > 0] = 255

        # tobecoveredarea = np.count_nonzero(board)
        # print('\t','Total area to be covered',tobecoveredarea)
        # print('\t','Area ratio',np.sum(trueareas)/tobecoveredarea)
        
        #plt.imshow(blurred_board)
        #plt.show()
        #cv2.imshow(filename,board)
        #cv2.waitKey()
        
        #print(sorted(tally.items()))
        
        
