#include "Heap.h"
#include <stdlib.h>

struct hp {
	int size, maxSize;
	Node *heap;
};

Heap HP_init(int size) {
	Heap newHeap = malloc(sizeof(Heap));
	newHeap->heap = malloc(size*sizeof(Node));
	newHeap->size = 0;
	newHeap->maxSize = size;
	return newHeap;
}

void HP_print(Heap h) {
	for(int i = 0; i<h->size; i++) {
		printf("%d\n", h->heap[i].cost);
	}
}

int HP_isEmpty(Heap h) { return !h->size;}

void HP_insert(Heap heap, int x, int y, int cost) {
	Node newNode = initNode(x,y,cost);
	int i = heap->size++;
	while(i>=1 && nodeCompare(newNode,heap->heap[parent(i)])<0) {
		heap->heap[i] = heap->heap[parent(i)];
		i = parent(i);
	}
	heap->heap[i] = newNode;
}

void HP_heapify(Heap h, int n) {
	int l, r, largest;
	Node swapNode;
	l = left(n);
	r = right(n);
	//if (l < heap->size && heap[l].n1>heap[i].n1)
	if (l < h->size && nodeCompare(h->heap[l],h->heap[n])<0)
		largest = l;
	else
		largest = n;
	if (r<h->size && nodeCompare(h->heap[r],h->heap[largest])<0)
		largest = r;
	if (largest != n) {
		swapNode = h->heap[n];
		h->heap[n] = h->heap[largest];
		h->heap[largest] = swapNode;
		HP_heapify(h,largest);
	}
}

int*HP_extractMax(Heap h) {
	int *v = malloc(2*sizeof(int));
	Node temp = h->heap[0];
	h->heap[0] = h->heap[h->size-1]; 
	
	v[0] = temp.x;
	v[1] = temp.y;
	//v[2] = temp.cost;
	h->size--;
	HP_heapify(h,0);
	
	return v;
}

Node initNode(int x, int y, int cost) {
	Node newNode;
	newNode.x = x;
	newNode.y = y;
	newNode.cost = cost;
	return newNode;
}

int nodeCompare(Node n1, Node n2) {return n1.cost - n2.cost;}
int left(int i) { return (i*2 + 1); }
int right(int i) { return (i*2 + 2); }
int parent(int i) { return ((i-1)/2); }