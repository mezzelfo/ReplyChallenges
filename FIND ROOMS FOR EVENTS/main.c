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


int main()
{
	FILE * inputFile;
	Event* EventsArray;
	Room* RoomsArray;
	unsigned eventsCount, roomsCount;

	inputFile = fopen("data_50000_100.in","r");
	if (inputFile == NULL)
	{
		fprintf(stderr, "Errore nell'apertura del file\n");
		return -1;
	}
	fscanf(inputFile, "%d %d", &eventsCount, &roomsCount);
	EventsArray = (Event*)malloc(sizeof(Event)*eventsCount);
	RoomsArray = (Room*)malloc(sizeof(Room)*roomsCount);
	if ((EventsArray == NULL) || (RoomsArray == NULL))
	{
		fprintf(stderr, "Errore nell'allocazione della memoria\n");
		return -2;
	}
	for (unsigned i = 0; i < eventsCount; ++i)
	{
		fscanf(inputFile, "%s %d %d %d",
			EventsArray[i].name,
			&EventsArray[i].start,
			&EventsArray[i].end,
			&EventsArray[i].participants);
		EventsArray[i].duration = EventsArray[i].end - EventsArray[i].start;
	}
	for (unsigned i = 0; i < roomsCount; ++i)
	{
		fscanf(inputFile, "%s %d",
			RoomsArray[i].name,
			&RoomsArray[i].capacity);
	}
	fclose(inputFile);




	free(EventsArray);
	free(RoomsArray);
	return 0;
}