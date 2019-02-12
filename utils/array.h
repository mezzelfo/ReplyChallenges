#ifndef ARRAY_H
#define ARRAY_H
#include <stdlib.h>

struct array
{
    void** ptr;
    unsigned size;
    unsigned capacity;
};

typedef struct array array;

array create_array();
void append(array* V, void* data);
void reserve(array* V, unsigned dim);
void trim(array* V);
void delete(array* V);

#endif