#include "list.h"

list create_list()
{
    list L;
    L.size = 0;
    L.head = (node*)malloc(sizeof(node));
    if (L.head == NULL) exit(EXIT_FAILURE);
    L.tail = (node*)malloc(sizeof(node));
    if (L.tail == NULL) exit(EXIT_FAILURE);
    L.head->data = NULL;
    L.head->perv = NULL;
    L.head->next = L.tail;
    L.tail->data = NULL;
    L.tail->perv = L.head;
    L.tail->next = NULL;
    return L;
}
void add_to_tail(list* L, void* data)
{
    node* new = (node*)malloc(sizeof(node));
    if (new == NULL) exit(EXIT_FAILURE);
    new->data = data;
    new->perv = L->tail->perv;
    L->tail->perv->next = new;
    new->next = L->tail;
    L->size++;
}
void add_to_head(list* L, void* data)
{
    node* new = (node*)malloc(sizeof(node));
    if (new == NULL) exit(EXIT_FAILURE);
    new->data = data;
    new->next = L->head->next;
    L->head->next->perv = new;
    new->perv = L->head;
    L->size++;
}