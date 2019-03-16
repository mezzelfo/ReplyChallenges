#ifndef HEAP
#define HEAP
//#include "Heap.c"
#include <stdio.h>
typedef struct hp *Heap;


typedef struct nd {
	int x, y, cost;
} Node;


Heap HP_init(int size);

void HP_insert(Heap heap, int x, int y, int cost);

int*HP_extractMax(Heap h);
int HP_isEmpty(Heap h);
void HP_print(Heap h);
Node initNode(int x, int y, int cost);
void HP_heapify(Heap h, int n);

int nodeCompare(Node n1, Node n2);
int left(int i);
int right(int i);
int parent(int i);


#endif