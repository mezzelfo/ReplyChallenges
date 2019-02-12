#include <stdlib.h>
#include <stdio.h>
#include "tool.h"

int main(int argc, char const **argv)
{
    Point startPoint, endPoint;
    unsigned obstaclesCount;
    Triangle* obstaclesArray;
    Point* graphNodes;
    unsigned nodesCount;
    float** CostMatrix;

    if (argc != 2)
    {
        fprintf(stderr,"Please use ./a.out <namefileinput>\n");
        exit(EXIT_FAILURE);
    }

    read_file(argv[1], &startPoint, &endPoint, &obstaclesArray, &obstaclesCount);
    is_problem_feasible(startPoint,endPoint,obstaclesArray,obstaclesCount);
    
    
    graphNodes = (Point*)calloc(4*3*obstaclesCount, sizeof(Point));
    nodesCount = 0;
    if (graphNodes == NULL)
    {
        fprintf(stderr,"Errore nell'allocazione della memoria\n");
        free(obstaclesArray);
        exit(EXIT_FAILURE);
    }
    for(Triangle* obs = obstaclesArray; obs != obstaclesArray + obstaclesCount; ++obs)
    {
        _Bool servono[] = {1,1,1,1,1,1,1,1,1,1,1,1};
        Point puntiVicini[] = { obs->a,obs->a,obs->a,obs->a,
                                obs->b,obs->b,obs->b,obs->b,
                                obs->c,obs->c,obs->c,obs->c};
        for (int i=0;i<2;i++)
        {
            puntiVicini[4*i+0].x ++;
            puntiVicini[4*i+1].y ++;
            puntiVicini[4*i+2].x --;
            puntiVicini[4*i+3].y --;
        }
        for(Triangle* tri = obstaclesArray; tri != obstaclesArray + obstaclesCount; ++tri)
        {
            for (int i=0;i<12;i++)
            {
                if (!servono[i] || point_is_in_triangle(puntiVicini+i, tri)) servono[i] = 0;
            }
        }
        for (int i=0;i<12;i++)
        {
            if (servono[i])
            {
                _Bool nuovo = 1;
                for(unsigned j = 0; j < nodesCount; j++)
                {
                    if ((graphNodes[j].x == puntiVicini[i].x) && (graphNodes[j].y == puntiVicini[i].y)) nuovo = 0;
                }
                if (nuovo)
                {
                    graphNodes[nodesCount] = puntiVicini[i];
                    nodesCount++;
                }
            }
        }
    }
    
    fprintf(stderr,"Punti grafo: \t%d\nPunti totali: \t%d\n", nodesCount, obstaclesCount*3);
    graphNodes = (Point*)realloc(graphNodes, (2+nodesCount)*sizeof(Point));
    nodesCount += 2; //StartPoint and EndPoint
    graphNodes[nodesCount - 1] = endPoint;
    graphNodes[nodesCount - 2] = startPoint;
    if (graphNodes == NULL) {
        fprintf(stderr,"Errore nell'allocazione della memoria\n");
        free(obstaclesArray);
        free(graphNodes);
        exit(EXIT_FAILURE);
    }
    CostMatrix = setup_square_matrix(nodesCount);

    for(unsigned idx1 = 0; idx1 < nodesCount; idx1++)
    {
        for(unsigned idx2 = idx1+1; idx2 < nodesCount; idx2++)
        {
            
            if (are_points_visible(graphNodes+idx1,graphNodes+idx2,obstaclesArray,obstaclesCount))
            {
                float distance = distance_ptp(graphNodes[idx1],graphNodes[idx2]);
                CostMatrix[idx1][idx2] = distance;
                CostMatrix[idx2][idx1] = distance;
            }
            else
            {
                CostMatrix[idx1][idx2] = INF;
                CostMatrix[idx2][idx1] = INF;
            }
            
        }
    }
    fprintf(stderr,"Ho finito di calcolare i pesi\n");
    int dim;
    int* x = dijkstra(CostMatrix, nodesCount-2, nodesCount-1, nodesCount, &dim);
    for(int i = 0; i < dim; i++)
    {
        printf("(%d,%d),", graphNodes[x[i]].x, graphNodes[x[i]].y);
    }
    

    free_square_matrix(CostMatrix, nodesCount);
    free(graphNodes);
    free(obstaclesArray);
    return 0;
}

