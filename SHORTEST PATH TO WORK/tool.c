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

int is_point_on_segment(const Point* P1, const Point* P2, const Point* A)
{
	// If you are here you should know that orientation_point_line returns 0
	int xmax = P1->x > P2->x ? P1->x : P2->x;
	int xmin = xmax == P1->x ? P2->x : P1->x;
	int ymax = P1->y > P2->y ? P1->y : P2->y;
	int ymin = ymax == P1->y ? P2->y : P1->y;
	if ((xmin > A->x) || (xmax < A->x)) return 0;
	if ((ymin > A->y) || (ymax < A->y)) return 0;
	return 1;
}

int is_point_in_triangle(const Point* P, const Triangle* T)
{
	int b1,b2,b3;
	b1 = orientation_point_line(&(T->A), &(T->B), P);
	if (b1 == 0)
	{
		if (is_point_on_segment(&(T->A), &(T->B), P))
			return 1;
		else
			return 0;
	}
	b2 = orientation_point_line(&(T->B), &(T->C), P);
	if (b2 == 0)
	{
		if (is_point_on_segment(&(T->B), &(T->C), P))
			return 1;
		else
			return 0;
	}
	b3 = orientation_point_line(&(T->C), &(T->A), P);
	if (b3 == 0)
	{
		if (is_point_on_segment(&(T->C), &(T->A), P))
			return 1;
		else
			return 0;
	}
	if ((b1==1)&&(b2==1)&&(b3==1)) return 1; //Is strictly inside
	return 0;
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
	Point P1,P2,P3;
	P1 = read_point(f);
	P2 = read_point(f);
	P3 = read_point(f);	

	Triangle T;
	T.A = P1;
	if((P2.y - P1.y)*(P3.x - P2.x) - (P3.y - P2.y)*(P2.x - P1.x) < 0)
	{
		T.B = P2;
		T.C = P3;
	}else
	{
		T.B = P3;
		T.C = P2;
	}
	return T;
}

void print_point(const Point P)
{
	printf("(%d,%d)", P.x, P.y);
}
