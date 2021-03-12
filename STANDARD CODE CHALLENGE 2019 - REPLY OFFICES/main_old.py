import numpy as np
import matplotlib.pyplot as plt
from skimage import graph as MCP

char_to_value = {'#':np.inf,'~':800.0,'*':200.0,'+':150.0,'X':120.0,'_':100.0,'H':70.0,'T':50.0}
move_to_value = {'U':(0,-1),'D':(0,+1),'L':(-1,0),'R':(+1,0)}

#with open('2_himalayas.txt','r') as file:
with open('1_victoria_lake.txt','r') as file:
    filecontent = file.readlines()

N,M,C,R = [int(x) for x in filecontent[0].strip().split()]
customer_hq_pos = [[int(x) for x in s.strip().split()] for s in filecontent[1:C+1]]
matrix = np.array([list(s.strip()) for s in filecontent[C+1:]])
matrix = np.vectorize(char_to_value.get)(matrix).transpose()

assert len(customer_hq_pos) == C
assert matrix.shape == (N,M)

for (x,y,rew) in customer_hq_pos:
    if matrix[x,y] == char_to_value['#']:
        print('Customer Headquarter on mountain')
    matrix[x,y] -= rew

class myMCP(MCP.MCP):
    def goal_reached(self,index,cumcost):
        if cumcost > 0: return 1
        return 0

MCP = myMCP(matrix,fully_connected=False)
startpos = [tuple(hq[:2]) for hq in customer_hq_pos]
cumcost, trace = MCP.find_costs(starts=startpos[:2],ends=startpos[:2])

print(trace)

plt.imshow(cumcost)
plt.scatter([x[1] for x in customer_hq_pos], [x[0] for x in customer_hq_pos], color='red')
plt.colorbar()
plt.show()