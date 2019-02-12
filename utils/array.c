#include "array.h"

array create_array()
{
    array V;
    V.size = 16;
    V.capacity = 16;
    V.ptr = malloc(V.capacity*sizeof(void*));
    if (V.ptr == NULL) exit(EXIT_FAILURE);
    return V;
}
void append(array* V, void* data)
{
    if (V->size + 1 >= V->capacity) {
        V->capacity *= 2;
        V->ptr = realloc(V->ptr, sizeof(void*)*(V->capacity));
        if (V->ptr == NULL) exit(EXIT_FAILURE);
    }
    int* tmp = V->ptr;
    tmp[(V->size)] = data;
    V->size++;   
}