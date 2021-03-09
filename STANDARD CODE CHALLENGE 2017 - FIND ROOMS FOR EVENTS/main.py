from decimal import *
class Event:
	def __init__(self, line):
		try:
			data = line.split(' ')
			self.name = data[0]
			self.start = int(data[1])
			self.end = int(data[2])
			self.duration = self.end - self.start
			self.people = int(data[3])
			self.partial_score = self.duration * self.people
			self.compatible_rooms = []
			self.compatible_next_events = []
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
with open('FIND ROOMS FOR EVENTS/data_50000_100.in','r') as file:
	eventCount, roomCount = [int(k) for k in file.readline().strip().split(' ')]
	for i in range(eventCount):
		line = file.readline().strip()
		events.append(Event(line))
	for i in range(roomCount):
		line = file.readline().strip()
		rooms.append(Room(line))
max_room_cap = max([r.places for r in rooms])
events = [e for e in events if e.people <= max_room_cap]

i = 1
for e in events:
	i *= len([r for r in rooms if r.places >= e.people])
getcontext().prec = 6
a = Decimal(i) / Decimal(1)
print(a)




print('Number of Hostable Events:',len(events))
print('Number of Rooms:',len(rooms))
print('Maximum room capacity',max_room_cap)

