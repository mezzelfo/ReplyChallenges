#include "array.h"

array create_array()
{
    array V;
    V.size = 0;
    V.capacity = 1;
    V.ptr = (void**)malloc(V.capacity*sizeof(void*));
    if (V.ptr == NULL) exit(EXIT_FAILURE);
    return V;
}
void append(array* V, void* data)
{
    if (V->size >= V->capacity) {
        V->capacity *= 2;
        V->ptr = realloc(V->ptr, sizeof(void*)*(V->capacity));
        if (V->ptr == NULL) exit(EXIT_FAILURE);
    }
    V->ptr[(V->size)] = data;
    V->size++;   
}
void reserve(array* V, unsigned dim)
{
    if (V->capacity < dim) {
        V->capacity = dim;
        V->ptr = realloc(V->ptr, sizeof(void*)*(V->capacity));
        if (V->ptr == NULL) exit(EXIT_FAILURE);
    }
}
void trim(array* V)
{
    V->capacity = V->size;
    V->ptr = realloc(V->ptr, sizeof(void*)*(V->capacity));
    if (V->ptr == NULL) exit(EXIT_FAILURE);
}
void delete(array* V)
{
    free(V->ptr);
}