#include "tool.h"

void read_file(const char* filename, Point* startPoint, Point* endPoint, Triangle** obstaclesArray, unsigned* obstaclesCount)
{
    FILE* inputFile;
    int checkScanf;
    unsigned i;
    
    inputFile = fopen(filename,"r");
    if (inputFile == NULL)
    {
        fprintf(stderr,"Errore nell'apertura del file di input\n");
        exit(EXIT_FAILURE);
    }
    checkScanf = fscanf(inputFile,"%d %d %d %d %u",&(startPoint->x),&(startPoint->y),&(endPoint->x),&(endPoint->y),obstaclesCount);
    if (checkScanf != 5)
    {
        fprintf(stderr,"Errore nella lettura del file di input\n");
        fclose(inputFile);
        exit(EXIT_FAILURE);
    }    
    *obstaclesArray = (Triangle*)malloc(*obstaclesCount * sizeof(Triangle));
    if (obstaclesArray == NULL)
    {
        fprintf(stderr,"Errore nell'allocazione della memoria\n");
        fclose(inputFile);
        exit(EXIT_FAILURE);
    }
    for(i = 0; i < *obstaclesCount; i++)
    {
        checkScanf = fscanf(inputFile,"%d %d %d %d %d %d",
                    &((*obstaclesArray)[i].a.x),
                    &((*obstaclesArray)[i].a.y),
                    &((*obstaclesArray)[i].b.x),
                    &((*obstaclesArray)[i].b.y),
                    &((*obstaclesArray)[i].c.x),
                    &((*obstaclesArray)[i].c.y)
                );
        if (checkScanf != 6)
        {
            fprintf(stderr,"Errore nella lettura del file di input\n");
            free(obstaclesArray);
            fclose(inputFile);
            exit(EXIT_FAILURE);
        }
    }
    fclose(inputFile);
}

void is_problem_feasible(const Point startPoint, const Point endPoint, const Triangle* obstaclesArray, const unsigned obstaclesCount)
{
    for(Triangle* i = obstaclesArray; i != obstaclesArray + obstaclesCount; ++i)
    {
        if (point_is_in_triangle(&startPoint, i) || point_is_in_triangle(&endPoint, i)) {
            fprintf(stderr,"StartPoint: (%d,%d)\n",startPoint.x,startPoint.y);
            fprintf(stderr,"EndPoint: (%d,%d)\n",endPoint.x,endPoint.y);
            fprintf(stderr,"Triangolo contenete uno dei due stremi: ((%d,%d)(%d,%d)(%d,%d))\n",
                            i->a.x,i->a.y,i->b.x,i->b.y,i->c.x,i->c.y);
            printf("IMPOSSIBLE\n");
            exit(EXIT_SUCCESS);
        }
    }
}


int sign(const Point* const p1, const Point* const p2, const Point* const p3)
{
  return (p1->x - p3->x) * (p2->y - p3->y) - (p2->x - p3->x) * (p1->y - p3->y);
}


_Bool point_is_in_triangle(const Point* const P, const Triangle* const T)
{
    _Bool b1, b2, b3;

    b1 = sign(P, &(T->a), &(T->b)) <= 0;
    b2 = sign(P, &(T->b), &(T->c)) <= 0;
    b3 = sign(P, &(T->c), &(T->a)) <= 0;

    return ((b1 == b2) && (b2 == b3));
}

int Side(const Point* p, const Point* q, const Point* a, const Point* b)
{
    int z1 = (b->x - a->x) * (p->y - a->y) - (p->x - a->x) * (b->y - a->y);
    int z2 = (b->x - a->x) * (q->y - a->y) - (q->x - a->x) * (b->y - a->y);
    return z1 * z2;
}

_Bool are_points_visible(const Point* p0, const Point* p1, Triangle* obstaclesArray, const unsigned obstaclesCount)
{
    for(Triangle* tri = obstaclesArray; tri != obstaclesArray+obstaclesCount; ++tri)
    {
        Point* t0 = &(tri->a);
        Point* t1 = &(tri->b);
        Point* t2 = &(tri->c);
       /* Check whether segment is outside one of the three half-planes
     * delimited by the triangle. */
    float f1 = Side(p0, t2, t0, t1), f2 = Side(p1, t2, t0, t1);
    float f3 = Side(p0, t0, t1, t2), f4 = Side(p1, t0, t1, t2);
    float f5 = Side(p0, t1, t2, t0), f6 = Side(p1, t1, t2, t0);
    /* Check whether triangle is totally inside one of the two half-planes
     * delimited by the segment. */
    float f7 = Side(t0, t1, p0, p1);
    float f8 = Side(t1, t2, p0, p1);

    /* If segment is strictly outside triangle, or triangle is strictly
     * apart from the line, we're not intersecting */
    if ((f1 < 0 && f2 < 0) || (f3 < 0 && f4 < 0) || (f5 < 0 && f6 < 0)
          || (f7 > 0 && f8 > 0)) continue;

    /* If segment is aligned with one of the edges, we're overlapping */
    if ((f1 == 0 && f2 == 0) || (f3 == 0 && f4 == 0) || (f5 == 0 && f6 == 0)) return 0;

    /* If segment is outside but not strictly, or triangle is apart but
     * not strictly, we're touching */
    if ((f1 <= 0 && f2 <= 0) || (f3 <= 0 && f4 <= 0) || (f5 <= 0 && f6 <= 0)
          || (f7 >= 0 && f8 >= 0)) return 0;

    /* If both segment points are strictly inside the triangle, we
     * are not intersecting either */
    if (f1 > 0 && f2 > 0 && f3 > 0 && f4 > 0 && f5 > 0 && f6 > 0) continue;

    /* Otherwise we're intersecting with at least one edge */
    return 0;
        

    }
    return 1;
}

float** setup_square_matrix(const unsigned size)
{
	float** Matrix = (float**)malloc(sizeof(float*)*size);
	if (Matrix == NULL) {
		fprintf(stderr, "Errore nell'allocazione della memoria\n");
		exit(EXIT_FAILURE);
	}
	for(unsigned i = 0; i < size; i++)
	{
		Matrix[i] = (float*)malloc(sizeof(float)*size);
		if (Matrix[i] == NULL) {
			fprintf(stderr, "Errore nell'allocazione della memoria\n");
			exit(EXIT_FAILURE);
		}
	}
	return Matrix;
}

void free_square_matrix(float** Matrix, const unsigned size)
{
	for(unsigned i = 0; i < size; i++) free(Matrix[i]);
	free(Matrix);	
}

int* dijkstra(float** cost, int source, int target, unsigned N, int* resultDim)
{
    int dist[N];
    int prev[N];
    int selected[N];
    for(unsigned i = 0; i < N; i++) selected[i] = 0;
    
    int i,m,min,start,d,j;
    int path[N];
    for(i=1;i< N;i++)
    {
        dist[i] = INF;
        prev[i] = -1;
    }
    start = source;
    selected[start]=1;
    dist[start] = 0;
    while(selected[target] ==0)
    {
        min = INF;
        m = 0;
        for(i=1;i< N;i++)
        {
            d = dist[start] +cost[start][i];
            if(d< dist[i]&&selected[i]==0)
            {
                dist[i] = d;
                prev[i] = start;
            }
            if(min>dist[i] && selected[i]==0)
            {
                min = dist[i];
                m = i;
            }
        }
        start = m;
        selected[start] = 1;
    }
    start = target;
    j = 0;
    while(start != -1)
    {
        path[j++] = start;
        start = prev[start];
    }
    
    *resultDim = j;
    int* resultPath = (int*)malloc(j*sizeof(int));
    for(int i = j-1; i >=0; i--)
    {
        resultPath[i] = path[i];
    }
    
    return resultPath;
}