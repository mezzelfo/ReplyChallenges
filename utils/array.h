#ifndef ARRAY_H
#define ARRAY_H
#include <stdlib.h>

struct array
{
    void* ptr;
    unsigned size;
    unsigned capacity;
};

typedef struct array array;

array create_array(unsigned init_size);
void append(array* V, void* data);

#endif