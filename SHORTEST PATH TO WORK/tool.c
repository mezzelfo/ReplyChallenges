#include "tool.h"

void* vec_alloc(unsigned size)
{
	void* p = malloc(size);
	if (p == NULL)
	{
		fprintf(stderr, "Problem encountered during memory allocation\n");
		exit(EXIT_FAILURE);
	}
	return p;
}

inline int prod_vect(const Point* P1, const Point* P2, const Point* A)
{
	return (P2->x - P1->x)*(A->y - P1->y)-(A->x - P1->x)*(P2->y - P1->y);
}

int orientation_point_line(const Point* P1, const Point* P2, const Point* A)
{
	if (prod_vect(P1,P2,A) > 0) return 1;	// Left
	if (prod_vect(P1,P2,A) < 0) return 2;	// Right
	return 0;								// On
}

int is_point_in_triangle(const Point* P, const Triangle* T)
{
	switch(orientation_point_line(&(T->A), &(T->B), P))
	{
		case 0:
			return 1;
		case 2:
			return 0;
		default:
			switch(orientation_point_line(&(T->B), &(T->C), P))
			{
				case 0:
					return 1;
				case 2:
					return 0;
				default:
					switch(orientation_point_line(&(T->B), &(T->C), P))
					{
						case 0:
							return 1;
						case 2:
							return 0;
						default:
							return 1;
					}
			}
	}
}

int read_int(FILE* f)
{
	int r;
	if (fscanf(f,"%d", &r) != 1)
	{
		fprintf(stderr, "Problem encountered during input file parsing\n");
		exit(EXIT_FAILURE);
	}
	return r;
}

Point read_point(FILE* f)
{
	Point P;
	P.x = read_int(f);
	P.y = read_int(f);
	return P;

}

Triangle read_triangle(FILE* f)
{
	Triangle T;
	T.A = read_point(f);
	T.B = read_point(f);
	T.C = read_point(f);
	return T;
}

void print_point(const Point P)
{
	printf("(%d,%d)", P.x, P.y);
}
