#include <stdlib.h>

int score(const int* const antennasPos, const int N, const int M, const int R, const int* const buildings,const int* const antennas)
{
    int score = 0;
    int testReward = 1;
    for (int i = 0; i < N; i++)
    {
        int connected = 0;
        int bestLocalScore = -9999999;
        for (int j = 0; j < N; j++)
        {
            int d = abs(buildings[4 * i] - antennasPos[2 * j]) + abs(buildings[4 * i + 1] - antennasPos[2 * j + 1]);
            if (d > antennas[2 * j])
            {
                continue;
            }
            connected = 1;
            int thisScore = buildings[4 * i + 3] * antennas[2 * j + 1] - buildings[4 * i + 2] * d;
            if (thisScore > bestLocalScore)
            {
                bestLocalScore = thisScore;
            }
        }
        if (connected == 0)
        {
            testReward = 0;
        }
        score += bestLocalScore;
    }
    return -(score + testReward * R);
}

// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>

// int W, H, N, M, R;
// int *buildings;
// int *antennas;
// int *antennasPos;

// int main(int argc, char const *argv[])
// {
//     FILE *fp = fopen(argv[1], "r");
//     fscanf(fp, "%d %d\n", &W, &H);
//     fscanf(fp, "%d %d %d\n", &N, &M, &R);
//     printf("%d %d %d %d %d\n", W, H, N, M, R);
//     buildings = (int *)malloc(sizeof(int) * N * 4);
//     antennas = (int *)malloc(sizeof(int) * M * 2);
//     antennasPos = (int *)malloc(sizeof(int) * M * 2);
//     for (int i = 0; i < N; i++)
//     {
//         fscanf(fp, "%d %d %d %d\n", &(buildings[4 * i]), &(buildings[4 * i + 1]), &(buildings[4 * i + 2]), &(buildings[4 * i + 3]));
//     }
//     for (int j = 0; j < M; j++)
//     {
//         fscanf(fp, "%d %d\n", &(antennas[2 * j]), &(antennas[2 * j + 1]));
//     }
//     srand(0);
//     for (int t = 0; t < 50; t++)
//     {
//         for (int j = 0; j < M; j++)
//         {
//             antennasPos[2 * j] = rand() % W;
//             antennasPos[2 * j + 1] = rand() % H;
//             //printf("(%d %d)\n", antennasPos[2 * j],antennasPos[2 * j+1]);
//         }
//         //printf("%d\n",score());
//     }
//     antennasPos[0] = 10;
//     antennasPos[1] = 7;
//     antennasPos[2] = 12;
//     antennasPos[3] = 2;
//     antennasPos[4] = 2;
//     antennasPos[5] = 4;
//     antennasPos[6] = 0;
//     antennasPos[7] = 7;
//     printf("%d\n",score());
//     return 0;
// }
