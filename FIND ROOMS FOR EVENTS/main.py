import networkx as nx

class Event:
	def __init__(self, line):
		try:
			data = line.split(' ')
			self.name = data[0]
			self.start = int(data[1])
			self.end = int(data[2])
			self.duration = self.end - self.start
			self.people = int(data[3])
		except:
			print(line)
			raise

class Room:
	def __init__(self, line):
		try:
			data = line.split(' ')
			self.name = data[0]
			self.places = int(data[1])
		except:
			print(line)
			raise

def unallocatedTime(schedule):
	return ((schedule[-1].end - schedule[0].start)-sum([e.duration for e in schedule]))

def score(schedule, room):
	s = sum([e.people*e.duration/room.places for e in schedule])
	s -= (room.places/max_room_cap)*unallocatedTime(schedule)
	return s

events = []
rooms = []
with open('FIND ROOMS FOR EVENTS/data_50000_10.in','r') as file:
	eventCount, roomCount = [int(k) for k in file.readline().strip().split(' ')]
	for i in range(eventCount):
		line = file.readline().strip()
		events.append(Event(line))
	for i in range(roomCount):
		line = file.readline().strip()
		rooms.append(Room(line))
max_room_cap = max([r.places for r in rooms])
events = [e for e in events if e.people <= max_room_cap and e.duration > 0]
emptyrooms = [r for r in rooms if r.places == 0]
rooms = sorted([r for r in rooms if r.places > 0], key=lambda x:x.places)
events.sort(key = lambda e: e.end)
print('Number of Hostable Events:',len(events))
print('Number of Rooms:',len(rooms))
print('Number of empty rooms', len(emptyrooms))
print('Maximum room capacity',max_room_cap)

longest = {}
used = []
total_score = 0
for r in reversed(rooms):
	G = nx.DiGraph()
	for e1 in events:
		if (e1.people <= r.places) and (e1 not in used):
			for e2 in events:
				if e2.people <= r.places and e1.end <= e2.start and e2 not in used:
					delta = (e2.people/r.places)*(e2.end - e1.start)/(e2.start - e1.end + 1)
					G.add_edge(e1,e2, weight = delta)

	print('DAG?',nx.is_directed_acyclic_graph(G))	
	p = nx.dag_longest_path(G, default_weight=-1)
	for e in p:
		used.append(e)
	print('Room',r.name,'Score:',score(p,r))
	total_score += score(p,r)
	longest[r] = p

print('Total Score',total_score)