#include <stdlib.h>
#include <stdio.h>
#include "tool.h"
#include "Dijkstra.h"

unsigned map_width, map_height, customer_num, max_reply_num;

typedef struct 
{
	unsigned x,y,reward;
	int rewardCosted;
} customer;

unsigned charToCost(const char c)
{
	switch (c)
	{
		case '#': return 999999999;
		case '~': return 800;
		case '*': return 200;
		case '+': return 150;
		case 'X': return 120;
		case '_': return 100;
		case 'H': return 70;
		case 'T': return 50;
	}
	printf("Ascii Carattere letto: %d\n", c);
	throwError("Errore nella lettura del carattere");
	return -1;
}

char CostToChar(const unsigned c)
{
	switch (c)
	{
		case 999999999: return '#';
		case 800: return '~';
		case 200: return '*';
		case 150: return '+';
		case 120: return 'X';
		case 100: return '_';
		case 70: return 'H';
		case 50: return 'T';
	}
	printf("Ascii Carattere letto: %d\n", c);
	throwError("QUIErrore nella lettura del carattere");
	return 1;
}

int cmpfunc (const void * a, const void * b) {
	customer* aa = (customer*)a;
	customer* bb = (customer*)b;
	return -( aa->rewardCosted - bb->rewardCosted );
}

typedef struct
{
	unsigned x,y;
} reply;

int main(int argc, char const *argv[])
{
	customer* customer_vector;
	reply* replyvector;
	unsigned** map;
	FILE* finput;
	FILE* foutput;

	if (argc != 2)
		throwError("Errore nel numero di argomenti. Usa ./a.out nomeinputfile");

	finput = fopen(argv[1],"r");
	if (finput == NULL)
		throwError("Errore. NOn sono riuscito ad aprire il file");

	map_width = readUnsigned(finput);
	map_height = readUnsigned(finput);
	customer_num = readUnsigned(finput);
	max_reply_num = readUnsigned(finput);

	replyvector = (reply*)secMalloc(sizeof(reply)*max_reply_num);

	customer_vector = (customer*)secMalloc(sizeof(customer)*customer_num);
	for(int i = 0; i < customer_num; i++)
	{
		customer_vector[i].x = readUnsigned(finput);
		customer_vector[i].y = readUnsigned(finput);
		customer_vector[i].reward = readUnsigned(finput);
	}

	map = (unsigned**)secMalloc(sizeof(unsigned*)*map_width);
	for(int i = 0; i < map_width; i++)
	{
		map[i] = (unsigned*)secMalloc(sizeof(unsigned)*map_height);
		for(int j = 0; j < map_height; j++)
			map[i][j] = charToCost(readChar(finput));
	}

	fclose(finput);


	foutput = fopen("out.txt","w");
	for (int i = 0; i < customer_num; ++i)
	{
		unsigned x = customer_vector[i].x;
		unsigned y = customer_vector[i].y;
		customer_vector[i].rewardCosted = customer_vector[i].reward - map[x][y];
		printf("(%u,%u)->%c %d\n", x,y,CostToChar(map[x][y]),customer_vector[i].rewardCosted);
	}
	qsort(customer_vector, customer_num, sizeof(customer), cmpfunc);
	int centriusati = 0;
	for(int i=0; i < customer_num; i++)
	{
		if(centriusati >= max_reply_num) break;

		unsigned x,y;
		if(customer_vector[i].rewardCosted < 0)
		{
			printf("Tutti gli altri customer sono inutili\n");
			break;
		}
		// NORD
		if (customer_vector[i].x > 0)
		{
			x = customer_vector[i].x-1;
			y = customer_vector[i].y;
			if(map[x][y] != charToCost('#'))
			{
				centriusati++;
				fprintf(foutput,"%u %u D\n", x,y);
				replyvector[centriusati].x = x;
				replyvector[centriusati].y = y;
			}

			if(centriusati >= max_reply_num) break;
		}
		

		if (customer_vector[i].x <= map_height-2)
		{
			// SUD
			x = customer_vector[i].x+1;
			y = customer_vector[i].y;
			if(map[x][y] != charToCost('#'))
			{
				centriusati++;
				fprintf(foutput,"%u %u U\n", x,y);
				replyvector[centriusati].x = x;
				replyvector[centriusati].y = y;
			}

			if(centriusati >= max_reply_num) break;
		}


		if (customer_vector[i].y <= map_width-2)
		{
			// EST
			x = customer_vector[i].x;
			y = customer_vector[i].y+1;
			if(map[x][y] != charToCost('#'))
			{
				centriusati++;
				fprintf(foutput,"%u %u L\n", x,y);
				replyvector[centriusati].x = x;
				replyvector[centriusati].y = y;
			}

			if(centriusati >= max_reply_num) break;

		}
		

		if (customer_vector[i].y > 0)
		{
			// OVEST
			x = customer_vector[i].x;
			y = customer_vector[i].y-1;
			if(map[x][y] != charToCost('#'))
			{
				centriusati++;
				fprintf(foutput,"%u %u R\n", x,y);
				replyvector[centriusati].x = x;
				replyvector[centriusati].y = y;
			}

			if(centriusati >= max_reply_num) break;
		}
	}

	unsigned*** ptr;
	for (int i = 0; i < centriusati; ++i)
	{
		ptr = shortestPaths(replyvector[i].x,replyvector[i].y,map,map_height,map_width);
		for(int j = 0; j < customer_num; j++)
		{
			if(ptr[0][customer_vector[j].x][customer_vector[j].y] < customer_vector.reward)
			{
				
			}

		}
	}
	

	free(replyvector);
	free(customer_vector);
	for(int i=0; i < map_width; i++) free(map[i]);
	free(map);

	return 0;
}