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

int main()
{
	FILE * inputFile;
	Event* EventsArray;
	Event* TopologicallyOrderedEventsArray;
	Room* RoomsArray;
	unsigned EventsCount, RoomsCount;
	unsigned max_capacity;
	float** WeightMatrix;

	inputFile = fopen("data_5000_3.in","r");
	if (inputFile == NULL)
	{
		fprintf(stderr, "Errore nell'apertura del file\n");
		exit(EXIT_FAILURE);
	}
	fscanf(inputFile, "%d %d", &EventsCount, &RoomsCount);
	EventsArray = (Event*)malloc(sizeof(Event)*EventsCount);
	RoomsArray = (Room*)malloc(sizeof(Room)*RoomsCount);
	if ((EventsArray == NULL) || (RoomsArray == NULL))
	{
		fprintf(stderr, "Errore nell'allocazione della memoria\n");
		free(EventsArray);
		free(RoomsArray);
		fclose(inputFile);
		exit(EXIT_FAILURE);
	}
	for (unsigned i = 0; i < EventsCount; ++i)
	{
		fscanf(inputFile, "%s %d %d %d",
			EventsArray[i].name,
			&EventsArray[i].start,
			&EventsArray[i].end,
			&EventsArray[i].participants);
		EventsArray[i].duration = EventsArray[i].end - EventsArray[i].start;
	}
	for (unsigned i = 0; i < RoomsCount; ++i)
	{
		fscanf(inputFile, "%s %d",
			RoomsArray[i].name,
			&RoomsArray[i].capacity);
	}
	fclose(inputFile);

	max_capacity = get_max_capacity(RoomsArray, RoomsCount);
	clean_rooms(&RoomsArray, &RoomsCount);
	clean_events(&EventsArray, &EventsCount, max_capacity);
	WeightMatrix = setup_square_matrix(EventsCount);

	for(unsigned idxE1 = 0; idxE1 < EventsCount; idxE1++)
	{
		for(unsigned idxE2 = 0; idxE2 < EventsCount; idxE2++)
		{
			const Event E1 = EventsArray[idxE1];
			const Event E2 = EventsArray[idxE2];
			if (E1.end <= E2.start) {
				/* (e2.people/r.places)*(e2.end - e1.start)/(e2.start - e1.end + 1) */
				WeightMatrix[idxE1][idxE2] = 1 + ((float)(E2.end - E1.start))/(E2.start - E1.end +1);
			}	
		}
	}
	TopologicallyOrderedEventsArray = topological_sort(EventsArray, WeightMatrix, EventsCount);

	free(EventsArray);
	free(TopologicallyOrderedEventsArray);
	free(RoomsArray);
	free_square_matrix(WeightMatrix, EventsCount);
	return 0;
}

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
