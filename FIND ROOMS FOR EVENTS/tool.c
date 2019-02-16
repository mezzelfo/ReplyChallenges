#include "tool.h"


unsigned get_max_capacity(const Room* const RoomsArray, const unsigned RoomsCount)
{
	unsigned max = 0;
	for(unsigned i = 0; i < RoomsCount; i++)
	{
		if (RoomsArray[i].capacity > max) max = RoomsArray[i].capacity;
	}
	return max;
}

void clean_rooms(Room** RoomsArray, unsigned* RoomsCount)
{
	unsigned CleanRoomsCount = 0;
	Room* CleanRoomsArray = (Room*)malloc(sizeof(Room)*(*RoomsCount));
	if (CleanRoomsArray == NULL) {
		fprintf(stderr, "Errore nell'allocazione della memoria\n");
		exit(EXIT_FAILURE);
	}
	for(unsigned i = 0; i < *RoomsCount; i++)
	{
		Room r = (*RoomsArray)[i];
		if (r.capacity > 0)
		{
			CleanRoomsArray[CleanRoomsCount] = r;
			CleanRoomsCount++;
		}
	}
	free(*RoomsArray);
	*RoomsArray = (Room*)realloc(CleanRoomsArray, CleanRoomsCount*sizeof(Room));
	*RoomsCount = CleanRoomsCount;
}

void clean_events(Event** EventsArray, unsigned* EventsCount, const unsigned max_capacity)
{
	unsigned CleanEventsCount = 0;
	Event* CleanEventsArray = (Event*)malloc(sizeof(Event)*(*EventsCount));
	if (CleanEventsArray == NULL) {
		fprintf(stderr, "Errore nell'allocazione della memoria\n");
		exit(EXIT_FAILURE);
	}	
	for(unsigned i = 0; i < *EventsCount; i++)
	{
		Event e = (*EventsArray)[i];
		if ((e.participants <= max_capacity) && (e.duration > 0))
		{
			CleanEventsArray[CleanEventsCount] = e;
			CleanEventsCount++;
		}
	}
	free(*EventsArray);
	*EventsArray = (Event*)realloc(CleanEventsArray, CleanEventsCount*sizeof(Event));
	*EventsCount = CleanEventsCount;
}

float** setup_square_matrix(const unsigned size)
{
	float** Matrix = (float**)malloc(sizeof(float*)*size);
	if (Matrix == NULL) {
		fprintf(stderr, "Errore nell'allocazione della memoria\n");
		exit(EXIT_FAILURE);
	}
	for(unsigned i = 0; i < size; i++)
	{
		Matrix[i] = (float*)malloc(sizeof(float)*size);
		if (Matrix[i] == NULL) {
			fprintf(stderr, "Errore nell'allocazione della memoria\n");
			exit(EXIT_FAILURE);
		}
	}
	return Matrix;
}

void free_square_matrix(float** Matrix, const unsigned size)
{
	for(unsigned i = 0; i < size; i++) free(Matrix[i]);
	free(Matrix);	
}

Event* topological_sort(const Event* const EventsArray, float** Matrix, const unsigned EventsCount)
{
	Event* ordered = (Event*)calloc(EventsCount, sizeof(Event));
	Flag* FlagsArray = (Flag*)calloc(EventsCount, sizeof(Flag));
	for(unsigned i = 0; i < EventsCount; i++)
	{
		if (FlagsArray[i] == NoFlag)
		{
			visit(i, FlagsArray, Matrix, EventsCount, ordered, EventsArray);
			i = 0;
		}
	}

	free(FlagsArray);
	return ordered;
}

void visit(const unsigned idxNode, Flag* FlagsArray, float** Matrix, const unsigned EventsCount, Event* L, const Event* const EventsArray)
{
	if (FlagsArray[idxNode] == PermFlag) return;
	if (FlagsArray[idxNode] == TempFlag)
	{
		fprintf(stderr, "Errore. Il grafo non è un DAG\n");
		exit(EXIT_FAILURE);
	}
	FlagsArray[idxNode] = TempFlag;
	for(unsigned i = 0; i < EventsCount; i++)
	{
		if (Matrix[idxNode][i] != 0) visit(i, FlagsArray, Matrix, EventsCount, L, EventsArray);
	}
	FlagsArray[idxNode] = PermFlag;
	for(unsigned i = EventsCount-1; i > 0; i--)
	{
		if (L[i].duration == 0) L[i] = EventsArray[i];
	}	
}
