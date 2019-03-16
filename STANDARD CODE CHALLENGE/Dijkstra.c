#include "Dijkstra.h"

unsigned int*** shortestPaths(int sourceX, int sourceY, unsigned int** map, int height, int width)
{
	//Up: 0
	//Right: 1
	//Down: 2
	//Left: 3
	unsigned int*** out=malloc(2*sizeof(unsigned int**));
	unsigned int** paths=malloc2d(width, height);
	unsigned int** extimates=malloc2d(width, height);
	int i, j, x, y;
	Heap PQ=HP_init(width*height);
	for(i=0; i<height; i++)
		for(j=0; j<width; j++)
			extimates[i][j]=999999999;
	int* maxNode=malloc(2*sizeof(int));
	HP_insert(PQ, sourceX, sourceY, map[sourceX][sourceY]);
	extimates[sourceX][sourceY]=0;
	paths[sourceX][sourceY]=-1;
	while(!HP_isEmpty(PQ))
	{
		maxNode=HP_extractMax(PQ);
		x=maxNode[0];
		y=maxNode[1];
		//Up
		if(x>0)
		{
			if(extimates[x][y]+map[x-1][y]<extimates[x-1][y])
			{
				extimates[x-1][y]=extimates[x][y]+map[x-1][y];
				paths[x-1][y]=0;
				HP_insert(PQ, x-1, y, map[x-1][y]);
			}
		}
		//Right
		if(y<width-1)
		{
			if(extimates[x][y]+map[x][y+1]<extimates[x][y+1])
			{
				extimates[x][y+1]=extimates[x][y]+map[x][y+1];
				paths[x][y+1]=1;
				HP_insert(PQ, x, y+1, map[x][y+1]);
			}
		}
		//Down
		if(x<height-1)
		{
			if(extimates[x][y]+map[x+1][y]<extimates[x+1][y])
			{
				extimates[x+1][y]=extimates[x][y]+map[x+1][y];
				paths[x+1][y]=2;
				HP_insert(PQ, x+1, y, map[x+1][y]);
			}
		}
		//Left
		if(y>0)
		{
			if(extimates[x][y]+map[x][y-1]<extimates[x][y-1])
			{
				extimates[x][y-1]=extimates[x][y]+map[x][y-1];
				paths[x][y-1]=3;
				HP_insert(PQ, x, y-1, map[x][y-1]);
			}
		}
	}
	out[0]=extimates;
	out[1]=paths;
	return out;
}

unsigned int** malloc2d(int width, int height)
{
	unsigned int** matr=malloc(height*sizeof(unsigned int*));
	int i;
	for(i=0; i<height; i++)
		matr[i]=malloc(width*sizeof(unsigned int));
	return matr;
}

char* getPath(unsigned int** paths, int height, int width, int destX, int destY)
{
	char* path=malloc(height*width*sizeof(char));
	getPathR(paths, path, destX, destY, -1);
	return path;
}

int getPathR(unsigned int** paths, char* path, int destX, int destY, int i)
{
	if(paths[destX][destY]!=-1)
	{
		if(paths[destX][destY]==0)
		{
			i=getPathR(paths, path, destX+1, destY, i);
			path[i]='U';
		}
		else if(paths[destX][destY]==1)
		{
			i=getPathR(paths, path, destX, destY-1, i);
			path[i]='R';
		}
		else if(paths[destX][destY]==2)
		{
			i=getPathR(paths, path, destX-1, destY, i);
			path[i]='D';
		}
		else if(paths[destX][destY]==0)
		{
			i=getPathR(paths, path, destX, destY+1, i);
			path[i]='L';
		}
	}
	return i+1;
}