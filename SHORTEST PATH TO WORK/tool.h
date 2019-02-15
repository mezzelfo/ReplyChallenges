#ifndef TOOL_H
#define TOOL_H
#include <stdio.h>
#include <stdlib.h>


// Structures
typedef struct {int x,y;} Point;
typedef struct {Point A,B,C;} Triangle;

// Generic helper functions
void* vec_alloc(unsigned size);
inline int prod_vect(const Point* P1, const Point* P2, const Point* A); //(P2-P1)x(A-P1)

// IO related functions
int read_int(FILE* f);
Point read_point(FILE* f);
Triangle read_triangle(FILE* f); // All triangle should be memorized in "anticlockwise mode"
void print_point(const Point P);

// Geometry related functions
int orientation_point_line(const Point* P1, const Point* P2, const Point* A);
int is_point_in_triangle(const Point* P, const Triangle* T);

#endif
