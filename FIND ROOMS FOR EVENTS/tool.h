#ifndef TOOL_H
#define TOOL_H

#include <stdio.h>
#include <stdlib.h>

typedef struct 
{
	char name[90];
	unsigned capacity;
} Room;

typedef struct 
{
	char name[90];
	unsigned start, end, participants;
	unsigned duration;
} Event;

unsigned get_max_capacity(const Room* const RoomsArray, const unsigned roomsCount);
void clean_rooms(Room** RoomsArray, unsigned* roomsCount);
void clean_events(Event** EventsArray, unsigned* eventsCount, const unsigned max_capacity);
float** setup_square_matrix(const unsigned size);
void free_square_matrix(float** Matrix, const unsigned size);

typedef enum {NoFlag, TempFlag, PermFlag} Flag;
Event* topological_sort(const Event* const EventsArray, float** Matrix, const unsigned EventsCount);
void visit(const unsigned idxNode, Flag* FlagsArray, float** Matrix, const unsigned EventsCount, Event* L, const Event* const EventsArray);


#endif
