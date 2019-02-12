#ifndef TOOL_H
#define TOOL_H
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define INF 9999999.0

typedef struct {int x,y;} Point;
typedef struct {Point a,b,c;} Triangle;

int Sign(const Point* const p1, const Point* const p2, const Point* const p3);
_Bool point_is_in_triangle(const Point* const P, const Triangle* const T);
_Bool are_points_visible(const Point* P1, const Point* P2, Triangle* obstaclesArray, const unsigned obstaclesCount);
float** setup_square_matrix(const unsigned size);
void free_square_matrix(float** Matrix, const unsigned size);

inline float distance_ptp(Point P1, Point P2)
{
    return sqrt((P1.x-P2.x)*(P1.x-P2.x)+(P1.y-P2.y)*(P1.y-P2.y));
}

int* dijkstra(float** cost, int source, int target, unsigned N, int* resultDim);

#endif