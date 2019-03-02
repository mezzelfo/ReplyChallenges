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
	int check;

	inputFile = fopen("data_50000_10.in","r");
	if (inputFile == NULL)
	{
		fprintf(stderr, "Errore nell'apertura del file\n");
		exit(EXIT_FAILURE);
	}
	check = fscanf(inputFile, "%u %u", &EventsCount, &RoomsCount);
	if (check!=2)
	{
		fprintf(stderr, "Errore nella lettura formattata del file\n");
		exit(EXIT_FAILURE);
	}
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
		check = fscanf(inputFile, "%s %u %u %u",
			EventsArray[i].name,
			&EventsArray[i].start,
			&EventsArray[i].end,
			&EventsArray[i].participants);
		if ((EventsArray[i].end < EventsArray[i].start)||(check!=4))
		{
			fprintf(stderr, "Errore nella lettura formattata del file\n");
			free(EventsArray);
			free(RoomsArray);
			fclose(inputFile);
			exit(EXIT_FAILURE);
		}
		EventsArray[i].duration = EventsArray[i].end - EventsArray[i].start;
	}
	for (unsigned i = 0; i < RoomsCount; ++i)
	{
		check = fscanf(inputFile, "%s %u",
			RoomsArray[i].name,
			&RoomsArray[i].capacity);
		if (check!=2)
		{
			fprintf(stderr, "Errore nella lettura formattata del file\n");
			exit(EXIT_FAILURE);
		}
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
