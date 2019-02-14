from shapely.geometry import Point, LineString, Polygon
def plane(a,b,c,d):
	for x in range(a,c+1):
		for y in range(b,d+1):
			yield (x,y)

triangle = Polygon([(-6,0),(0,-12),(6,0)])
start = Point(-10,0)
end = Point(10,0)

paths = []
for p1 in plane(-2,-10,10,2):
	path = LineString([start,p1,end])
	if triangle.disjoint(path):
		paths.append(path)
	for p2 in plane(-2,-10,10,2):
		path = LineString([start,p1,p2,end])
		if triangle.disjoint(path) and p1 != p2:
			paths.append(path)

paths.sort(key = lambda x:x.length)
print('Più corto:',list(paths[0].coords),paths[0].length)
print([(list(paths[0].coords),paths[0].length) for path in paths[:5]])