import matplotlib.pyplot as plt
from math import sqrt, ceil
from sys import getsizeof
import numpy


def distance(p1,p2):
	return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

class Triangle:
	def __init__(self, line):
		a,b,c,d,e,f = [int(k) for k in line.strip().split(' ')]
		self.vertices = [(a,b),(c,d),(e,f)]
		self.sides = [sqrt((a-c)**2+(b-d)**2),sqrt((a-e)**2+(b-f)**2),sqrt((c-e)**2+(d-f)**2)]
		sp = sum(self.sides)/2
		ar = sqrt(sp*(sp-self.sides[0])*(sp-self.sides[1])*(sp-self.sides[2]))
		try:
			x = -(-a**2*d+a**2*f-b**2*d+b**2*f+b*c**2+b*d**2-b*e**2-b*f**2-c**2*f-d**2*f+d*e**2+d*f**2)/(2*(a*d-b*c-a*f+b*e+c*f-d*e))
			y = (-a**2*c+a**2*e+a*c**2+a*d**2-a*e**2-a*f**2-b**2*c+b**2*e-c**2*e+c*e**2+c*f**2-d**2*e)/(2*(a*d-b*c-a*f+b*e+c*f-d*e))
		except ZeroDivisionError:
			print('Il triangolo è degenere:',self.vertices)
			xmin = min(a,c,e)
			xmax = max(a,c,e)
			x = (xmin+xmax)/2
			ymin = min(b,d,f)
			ymax = max(b,d,f)
			y = (ymin+ymax)/2
			self.circocentro = (x,y)
			self.circoraggio = max([distance(self.circocentro, v) for v in self.vertices])
			return
		self.circocentro = (x,y)
		self.circoraggio = self.sides[0]*self.sides[1]*self.sides[2]/(4*ar)


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
		obstacles.append(Triangle(line))

if num != len(obstacles):
	raise ValueError('Lettura del file non corretta')

print('Ho letto e memorizzato',num,'ostacoli')
print('Punto inizio:',start)
print('Punto fine:',end)

traiettoria = [(1999, 1999),(2000, 2000),(1900, 2001),(1543, 1691),(610, 1120),(510, 755),(388, 434),(138, 252)]
print('Preparo il grafico')
i = 0
plt.scatter(start[0],start[1],color='red')
plt.scatter(end[0],end[1],color='black')
plt.plot([k[0] for k in traiettoria], [k[1] for k in traiettoria], color='green')
for t in obstacles:
	i += 1
	if any([(min(start[0],end[0])<=k[0]<=max(start[0],end[0])) and (min(start[1],end[1])<=k[1]<=max(start[1],end[1]))  for k in t.vertices]):
		x = [k[0] for k in t.vertices]+[t.vertices[0][0]]
		y = [k[1] for k in t.vertices]+[t.vertices[0][1]]
		plt.fill(x,y,color='b')
		#print(i)
print('Sto per printare')
plt.show()