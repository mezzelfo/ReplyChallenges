import gurobipy as gp
from gurobipy import GRB
import numpy as np

def linetoint(line):
    return [int(n) for n in line.strip().split(' ')]


filename = 'data_scenarios_a_example.in'
#filename = 'data_scenarios_b_mumbai.in'

with open(filename) as f:
    print(filename)
    W,H = linetoint(f.readline())
    N,M,R = linetoint(f.readline())
    buildings = np.asarray([linetoint(f.readline()) for _ in range(N)])
    antennas = np.asarray([linetoint(f.readline()) for _ in range(M)])
print('Start modelling')

m = gp.Model("reply2021")


# positions = m.addVars(M,2, lb = 0, ub = [[H,W]]*M, name = 'positions')
# z = m.addVars(N,M, lb = -float('inf'), name = 'z')
# s = m.addVars(N, lb = -float('inf'), name = 's')
# for i in range(N):
#     m.addGenConstrMax(s[i], [z[i,j] for j in range(M)])
#     for j in range(M):
#         m.addConstr(z[i,j] <= buildings[i,3]*antennas[j,1] - buildings[i,2]*(((buildings[i,0]-positions[j,0])+(buildings[i,1]-positions[j,1]))))
#         m.addConstr(z[i,j] <= buildings[i,3]*antennas[j,1] - buildings[i,2]*((buildings[i,0]-positions[j,0])-(buildings[i,1]-positions[j,1])))
#         m.addConstr(z[i,j] <= buildings[i,3]*antennas[j,1] - buildings[i,2]*(-(buildings[i,0]-positions[j,0])+(buildings[i,1]-positions[j,1])))
#         m.addConstr(z[i,j] <= buildings[i,3]*antennas[j,1] - buildings[i,2]*(-(buildings[i,0]-positions[j,0])-(buildings[i,1]-positions[j,1])))
# m.setObjective(gp.quicksum(s[i] for i in range(N)), GRB.MAXIMIZE)

positions = m.addMVar(shape = (M,2) , lb = 0, name = 'positions')
norm1 = m.addMVar(shape = (N,2), lb = 0)
choosen = m.addMVar(shape = (N,M), lb = 0, ub = 1, vtype=GRB.BINARY)
bonus = m.addMVar(shape = (1), lb = 0, ub = 1, vtype=GRB.BINARY, name = 'bonus')

print(W,H)
m.addConstr(positions[:,0] <= W)
m.addConstr(positions[:,1] <= H)

m.addConstrs(bonus <= choosen[i,:].sum() for i in range(N))
m.addConstrs(choosen[i,:].sum() <= 1 for i in range(N))

m.addConstrs(norm1[i,j] >= (buildings[i,j]-(choosen[i,:] @ positions[:,j]))  for i in range(N) for j in range(2))
m.addConstrs(norm1[i,j] >= -(buildings[i,j]-(choosen[i,:] @ positions[:,j]))  for i in range(N) for j in range(2))

m.addConstrs( norm1[i,:].sum() <= choosen[i,:] @ antennas[:,0] for i in range(N))

obj1 = sum(buildings[i,3] * choosen[i,j] * antennas[j,1] for i in range(N) for j in range(M))
obj2 = sum(buildings[i,2] * norm1[i,j] for i in range(N) for j in range(2))
obj3 = R*bonus

m.setObjective(obj1-obj2+obj3, GRB.MAXIMIZE)
m.params.NonConvex = 2

print('Start solving')
m.optimize()
print('End Solving')
print('Obj: %g' % m.objVal)

print(M)
for j in range(M):
    posj0 = m.getVarByName(f'positions[{2*j}]').x
    posj1 = m.getVarByName(f'positions[{2*j+1}]').x
    #print(f'Antenna {j}: ({round(posj0)},{round(posj1)})')
    print(j,int(round(posj0)),int(round(posj1)))