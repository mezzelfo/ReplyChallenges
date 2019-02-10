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


def sign (p1,p2,p3):
	return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def pointInTriangle(t,p):
	d1 = sign(p, t.vertices[0],t.vertices[1])
	d2 = sign(p, t.vertices[1],t.vertices[2])
	d3 = sign(p, t.vertices[2],t.vertices[0])
	has_neg = (d1 <= 0) or (d2 <= 0) or (d3 <= 0)
	has_pos = (d1 >= 0) or (d2 >= 0) or (d3 >= 0)
	return not(has_neg and has_pos)

def connessi(t1,t2):
	if distance(t1.circocentro, t2.circocentro) > t1.circoraggio + t2.circoraggio:
		return False
	if any([intersecate(t1,p) for p in t2.vertices]):
		return True
	if any([intersecate(t2,p) for p in t1.vertices]):
		return True
	return False

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

#Controllo se i punti di partenza e di arrivo sono dentro dei triangoli
for t in obstacles:
	if distance(start,t.circocentro) < t.circoraggio:
		if(pointInTriangle(t,start)):
			print('Il punto',start,'è all\'interno di',t.vertices)
			exit()
	if distance(end,t.circocentro) < t.circoraggio:
		if(pointInTriangle(t,end)):
			print('Il punto',end,'è all\'interno di',t.vertices)
			exit()

noTouchy = []
i = 0
for t in obstacles:
	i+=1
	print(i)
	r = ceil(t.circoraggio)
	c = (ceil(t.circocentro[0]),ceil(t.circocentro[1]))
	for x in range(c[0]-r,c[0]+r):
		for y in range(c[1]-r,c[1]+r):
			if pointInTriangle(t,(x,y)):
				noTouchy.append((x,y))
print(getsizeof(noTouchy),'bytes')
print(len(noTouchy))
exit()

print('Preparo il grafico')
i = 0
plt.scatter(start[0],start[1],color='red')
plt.scatter(end[0],end[1],color='black')
for p in traiettoria:
	plt.scatter(p[0],p[1],color='green')
for t in obstacles:
	i += 1
	if any([(min(start[0],end[0])<=k[0]<=max(start[0],end[0])) and (min(start[1],end[1])<=k[1]<=max(start[1],end[1]))  for k in t.vertices]):
		x = [k[0] for k in t.vertices]+[t.vertices[0][0]]
		y = [k[1] for k in t.vertices]+[t.vertices[0][1]]
		plt.fill(x,y,color='b')
		#print(i)
print('Sto per printare')
plt.show()
