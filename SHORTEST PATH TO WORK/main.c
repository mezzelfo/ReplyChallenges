#include <stdio.h>
#include <stdlib.h>
#include "tool.h"

int main(int argc, char const *argv[])
{
	FILE* inputFile;
	Point startPoint, endPoint;
	int obsCount;
	Triangle* obsVec;


	if (argc != 2)
	{
		fprintf(stderr, "Please use ./a.out <inputfilename>\n");
		exit(EXIT_FAILURE);
	}
	if ((inputFile = fopen(argv[1],"r")) == NULL)
	{
		fprintf(stderr, "Problem encountered during input file opening\n");
		exit(EXIT_FAILURE);
	}
	startPoint = read_point(inputFile);
	endPoint = read_point(inputFile);
	obsCount = read_int(inputFile);
	obsVec = (Triangle*) vec_alloc(obsCount * sizeof(Triangle));
	for (int i = 0; i < obsCount; ++i)
		obsVec[i] = read_triangle(inputFile);

	for (Triangle* T = obsVec; T != obsVec+obsCount; ++T)
	{
		if (   is_point_in_triangle(&startPoint, T)
			|| is_point_in_triangle(&endPoint, T)
		   )
		{
			printf("IMPOSSIBLE\n");
			return 0;
		}
		
	}
		

	return 0;
}
