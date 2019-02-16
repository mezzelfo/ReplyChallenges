#include <stdio.h>
#include <stdlib.h>
#include "tool.h"

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
