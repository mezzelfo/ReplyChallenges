#include "Heap.h"
#include <stdlib.h>

unsigned int*** shortestPaths(int, int, unsigned int**, int, int);
unsigned int** malloc2d(int, int);
char* getPath(unsigned int**, int, int, int, int);
int getPathR(unsigned int**, char*, int, int, int);