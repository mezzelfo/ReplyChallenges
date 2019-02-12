import matplotlib.pyplot as plt
from math import sqrt, ceil
from sys import getsizeof
import numpy


def distance(p1,p2):
	return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def sign (p1,p2,p3):
	return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def pointInTriangle(t,p):
	d1 = sign(p, t[0],t[1])
	d2 = sign(p, t[1],t[2])
	d3 = sign(p, t[2],t[0])
	has_neg = (d1 <= 0) or (d2 <= 0) or (d3 <= 0)
	has_pos = (d1 >= 0) or (d2 >= 0) or (d3 >= 0)
	return not(has_neg and has_pos)

#main
obstacles = []
start = 0
end = 0
num = 0
with open('input_1.txt','r') as file:
	#read first line containing start and end point
	a,b,c,d = [int(k) for k in file.readline().strip().split(' ')]
	start = (a,b)
	end = (c,d)
	#read second line containing number of obstacles
	num = int(file.readline().strip())
	#read the rest of the file
	for line in file:
		a,b,c,d,e,f = [int(k) for k in line.strip().split(' ')]
		obstacles.append(((a,b),(c,d),(e,f)))

if num != len(obstacles):
	raise ValueError('Lettura del file non corretta')

print('Ho letto e memorizzato',num,'ostacoli')
print('Punto inizio:',start)
print('Punto fine:',end)

#Controllo se i punti di partenza e di arrivo sono dentro dei triangoli
for t in obstacles:
	if(pointInTriangle(t,start)):
		print('Il punto',start,'è all\'interno di',t)
		exit()
	if(pointInTriangle(t,end)):
		print('Il punto',end,'è all\'interno di',t)
		exit()

i = 0
graphNodes = set()
for triangle in obstacles:
	i +=1
	print(i)
	for v in triangle:
		for p in [(v[0]+1,v[1]),(v[0]-1,v[1]),(v[0],v[1]+1),(v[0],v[1]-1)]:
			if not any([pointInTriangle(t,p) for t in obstacles]):
				graphNodes.add(p)

print(len(graphNodes))