#ifndef LIST_H
#define LIST_H
#include <stdlib.h>

struct node
{
    void* data;
    struct node* next;
    struct node* perv;
};

struct list
{
    node* head;
    node* tail;
    unsigned size;
};

typedef struct list list;
typedef struct node node;

list create_list();
void add_to_tail(list* L, void* data);
void add_to_head(list* L, void* data);


#endif