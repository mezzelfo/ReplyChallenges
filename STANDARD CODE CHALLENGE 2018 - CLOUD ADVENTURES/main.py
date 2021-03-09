class Region:
	def __init__(self,regName,packNum,packCost,packPerServices,latency):
		self.name = str(regName)
		self.packNum = int(packNum)
		self.packCost = float(packCost)
		self.packPerServices = [int(k) for k in packPerServices]
		self.latency = [int(k) for k in latency]
		if '::' not in self.name:
			raise ValueError('name of region do not contain Provider name.')
		if len(self.packPerServices) != S:
			raise ValueError('len(packPerServices) is different from num of services.')
		if len(self.latency) != C:
			raise ValueError('len(latency) is different from num of country.')

class Project:
	def __init__(self, pen, country, units):
		self.penality = int(pen)
		self.country = countries[country]
		self.units = [int(k) for k in units]
		if len(self.units) != S:
			raise ValueError('len(units) is different from num of services.')


regions = []
projects = []
with open('fourth_adventure.in','r') as inputFile:
	V, S, C, P = [int(k) for k in inputFile.readline().strip().split(' ')]
	services = [k for k in inputFile.readline().strip().split(' ')]
	countriesNames = [k for k in inputFile.readline().strip().split(' ')]
	countries = {countriesNames[k]:k for k in range(len(countriesNames))}
	provNames = []
	for x in range(V):
		provName, r = [k for k in inputFile.readline().strip().split(' ')]
		provNames.append(provName)
		for y in range(int(r)):
			regName = inputFile.readline().strip() +'::'+provName
			data = [k for k in inputFile.readline().strip().split(' ')]
			latency = [int(k) for k in inputFile.readline().strip().split(' ')]
			regions.append(Region(regName,data[0],data[1],data[2:],latency))
	for x in range(P):
		data = [k for k in inputFile.readline().strip().split(' ')]
		projects.append(Project(data[0], data[1], data[2:]))

projects.sort(key = lambda p: sum(p.units))
