#ifndef SORTED_LINKED_LIST
#define SORTED_LINKED_LIST
#include <stdlib.h>

typedef struct PairStruct
{
    void* r1;
    void* r2;
    unsigned pow;
    struct PairStruct* next;
    struct PairStruct* prev;
} Pair;

typedef struct
{
    Pair* head;
    Pair* tail;
    Pair* iterator;
    unsigned size;
    unsigned MAX_SIZE;
} List;

List* getNewList(unsigned max)
{
    List* l = (List*)calloc(1,sizeof(List));
    l->head = NULL;
    l->tail = NULL;
    l->iterator = NULL;
    l->size = 0;
    l->MAX_SIZE = max;
    return l;
}

void clearList(List* l)
{
    while (l->head->next != NULL)
    {
        l->head = l->head->next;
        free(l->head->prev);
    }
    free(l->tail);
    
}

void addToList(List* l, void* r1, void* r2, unsigned pow)
{
    if(pow == 0) return;
    if(l->size == 0)
    {
        Pair* n = (Pair*)calloc(1,sizeof(Pair));
        n->next = NULL;
        n->prev = NULL;
        n->r1 = r1;
        n->r2 = r2;
        n->pow = pow;
        l->head = n;
        l->tail = n;
        l->size = 1;
        return;
    }
    if(l->head->pow < pow)
    {
        Pair* n = (Pair*)calloc(1,sizeof(Pair));
        n->next = NULL;
        n->prev = NULL;
        n->r1 = r1;
        n->r2 = r2;
        n->pow = pow;

        n->next = l->head;
        l->head->prev = n;
        l->head = n;
        l->size++;
    } else if(l->size < l->MAX_SIZE || l->tail->pow < pow)
    {
        Pair* n = (Pair*)calloc(1,sizeof(Pair));
        n->next = NULL;
        n->prev = NULL;
        n->r1 = r1;
        n->r2 = r2;
        n->pow = pow;

        Pair* here = l->head;
        while(here->next != NULL && here->next->pow > n->pow) here = here->next;
        //Inserisco n dopo here
        n->next = here->next;
        n->prev = here;
        if(here->next != NULL) here->next->prev = n; else l->tail = n;
        here->next = n;
        l->size++;        
    }
    if (l->size >= l->MAX_SIZE)
    {
        l->tail = l->tail->prev;
        free(l->tail->next);
        l->tail->next = NULL;
        l->size--;
    }
}

#endif