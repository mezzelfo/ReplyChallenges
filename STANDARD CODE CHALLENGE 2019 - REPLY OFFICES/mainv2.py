import numpy as np
import matplotlib.pyplot as plt
from skimage import graph
from os import listdir

char_to_value = {'#':np.inf,'~':800.0,'*':200.0,'+':150.0,'X':120.0,'_':100.0,'H':70.0,'T':50.0}
move_to_value = {'U':(0,-1),'D':(0,+1),'L':(-1,0),'R':(+1,0)}

class myMCP(graph.MCP):
    def goal_reached(self,index,cumcost):
        if cumcost > 0: return 1
        return 0

def fun(filecontent, filename):
    N,M,C,R = [int(x) for x in filecontent[0].strip().split()]
    customer_hq_pos = [[int(x) for x in s.strip().split()] for s in filecontent[1:C+1]]
    matrix = np.array([list(s.strip()) for s in filecontent[C+1:]])
    matrix = np.vectorize(char_to_value.get)(matrix)
    allmatrices = np.tile(matrix,(C,1,1))
    allcumcosts = np.zeros(allmatrices.shape)
    alltraceback = np.zeros(allmatrices.shape)

    assert len(customer_hq_pos) == C
    assert matrix.shape == (M,N)
    assert allmatrices.shape == (C,M,N)

    for i,(y,x,rew) in enumerate(customer_hq_pos):
        allmatrices[i,x,y] -= rew
        mcp = myMCP(allmatrices[i],fully_connected=False)
        cumcost, traceback = mcp.find_costs(starts=[(x,y)])
        allcumcosts[i] = cumcost
        alltraceback[i] = traceback
        
    sumcumcosts = np.ma.masked_invalid(allcumcosts).sum(axis=0)
    sumcumcosts = sumcumcosts.filled(1000)

    bestpos = np.unravel_index(np.argpartition(sumcumcosts.ravel(), R-1)[:R],sumcumcosts.shape)
    assert len(bestpos)==2 and len(bestpos[0])==len(bestpos[1]) and len(bestpos[0])==R
    print(bestpos)
    plt.figure()
    plt.title(filename)
    plt.scatter([x[0] for x in customer_hq_pos], [x[1] for x in customer_hq_pos], color='red')
    plt.scatter(bestpos[1], bestpos[0], color='yellow')
    plt.imshow(sumcumcosts,alpha=0.2)
    plt.colorbar()
    plt.show()

#for filename in listdir():
#    if '.txt' in filename:    
filename = '0_example.txt'
with open(filename,'r') as file:
    filecontent = file.readlines()
    fun(filecontent, filename)


        