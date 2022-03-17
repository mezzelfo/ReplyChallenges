// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

#include <iostream>
#include <fstream>
#include <stdio.h>

struct Antenna
{
    int range, speed, ID;
    int placed;
    int x, y;
};

double scoreAntenna(const Antenna &a)
{
    double x1 = 1.0 * a.range;
    double x2 = 1.0 * a.speed;
    //return (0.020213772296552)*x1+(9.287874055129072e-08)*x2+(9.206660025069180e-05)*x1*x2+x1*x1+(-2.426132491619204e-11)*x2*x2;
    //return x1+0.000001871996283*x2;
    return sqrt(x1) + x2;
}

long int *optimisticScore;
int compareAntennas(const void *a, const void *b)
{
    const Antenna *A = (Antenna *)a;
    const Antenna *B = (Antenna *)b;
    //return scoreAntenna(*A) < scoreAntenna(*B);
    return optimisticScore[B->ID] > optimisticScore[A->ID];
}

struct Building
{
    int x, y, latency, speed;
    int antenna_connected_to;
    int best_score;
    int second_best_score;
};

__global__ void solutionEval(Building *buildings, const Antenna *antennas, const int M, int *antennas_contrib_score)
{
    const int build_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const Building b = buildings[build_idx];
    int best_score = 0;
    int second_best_score = 0;
    int antenna_connected_to = -1;
    for (size_t a_idx = 0; a_idx < M; a_idx++)
    {
        const Antenna a = antennas[a_idx];
        if (a.placed)
        {
            int dist = abs(b.x - a.x);
            if (dist <= a.range)
            {
                dist += abs(b.y - a.y);
                if (dist <= a.range)
                {
                    const int contrib = b.speed * a.speed - b.latency * dist;
                    if ((contrib >= best_score) | (antenna_connected_to == -1))
                    {
                        best_score = contrib;
                        antenna_connected_to = a_idx;
                    }
                    else if (contrib > second_best_score)
                    {
                        second_best_score = contrib;
                    }
                }
            }
        }
    }
    buildings[build_idx].best_score = best_score;
    buildings[build_idx].second_best_score = second_best_score;
    buildings[build_idx].antenna_connected_to = antenna_connected_to;
    if (antenna_connected_to != -1)
    {
        atomicAdd(antennas_contrib_score + antenna_connected_to, best_score);
    }
}

__global__ void antennaPositionEval(const size_t a_idx, const Antenna a, const int H, const int W, const int N, const Building *buildings, const int *dev_buildings_map, int *score_map)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Devo decidere cosa succede quando (x,y) == (a.x,a.y)
    // Scelgo in quel caso score_map = 0

    int score = 0;
    for (int yy = max(0, y - a.range); yy <= min(H - 1, y + a.range); yy++)
    {
        const int dif = a.range - abs(y - yy);
        for (int xx = max(0, x - dif); xx <= min(W - 1, x + dif); xx++)
        {
            const int build_idx = dev_buildings_map[W * yy + xx];
            if (build_idx != -1)
            {
                const Building b = buildings[build_idx];
                const int dist = abs(y - yy) + abs(x - xx);
                const int contrib = b.speed * a.speed - b.latency * dist;
                // score += max(contrib, b.second_best_score) - b.best_score;
                if (b.antenna_connected_to != a_idx)
                {
                    if (contrib >= b.best_score)
                    {
                        score += contrib - b.best_score;
                    }
                    else if (b.antenna_connected_to == -1)
                    {
                        score += contrib;
                    }
                }
            }
        }
    }

    for (size_t b_idx = 0; b_idx < N; b_idx++)
    {
        if (buildings[b_idx].antenna_connected_to == a_idx)
        {
            const Building b = buildings[b_idx];
            const int dist = abs(y - b.y) + abs(x - b.x);
            if (dist > a.range)
            {
                score += b.second_best_score - b.best_score;
            }
            else
            {
                const int contrib = b.speed * a.speed - b.latency * dist;
                if (contrib >= b.best_score)
                {
                    score += contrib - b.best_score;
                }
                else if (contrib >= b.second_best_score)
                {
                    score += contrib - b.best_score;
                }
                else if (contrib < b.second_best_score)
                {
                    score += b.second_best_score - b.best_score;
                }
            }
        }
    }

    score_map[W * y + x] = score;
}

int main(int argc, char const *argv[])
{
    /* 
    create random solution
    load data on GPU

    for iteration {
        select an antenna (randomly?)
        get best position for that antenna
        move that antenna & adjust solution
    }
     */

    Building *buildings, *dev_buildings;
    Antenna *antennas, *dev_antennas;

    int *buildings_map, *score_map, *antennas_contrib_score;
    int *dev_buildings_map, *dev_score_map, *dev_antennas_contrib_score;

    int W, H, N, M, R;
    std::ifstream file(argv[1], std::ios::in);
    file >> W >> H >> N >> M >> R;

    cudaHostAlloc((void **)&buildings, N * sizeof(Building), cudaHostAllocDefault);
    cudaHostAlloc((void **)&antennas, M * sizeof(Antenna), cudaHostAllocDefault);
    cudaHostAlloc((void **)&antennas_contrib_score, M * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&(dev_buildings), N * sizeof(Building));
    cudaMalloc((void **)&(dev_antennas), M * sizeof(Antenna));
    cudaMalloc((void **)&(dev_antennas_contrib_score), M * sizeof(int));

    buildings_map = (int *)malloc(H * W * sizeof(int));
    score_map = (int *)malloc(H * W * sizeof(int));
    memset(buildings_map, -1, H * W * sizeof(int));

    cudaMalloc((void **)&(dev_buildings_map), H * W * sizeof(int));
    cudaMalloc((void **)&(dev_score_map), H * W * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        int x, y, l, s;
        file >> x >> y >> l >> s;
        buildings[i].x = x;
        buildings[i].y = y;
        buildings[i].latency = l;
        buildings[i].speed = s;
        buildings[i].antenna_connected_to = -1;
        buildings[i].best_score = INT_MIN;
        buildings_map[W * y + x] = i;
    }
    srand(time(NULL));
    for (int i = 0; i < M; i++)
    {
        int r, s;
        file >> r >> s;
        antennas[i].ID = i;
        antennas[i].range = r;
        antennas[i].speed = s;
        antennas[i].placed = 0;
        antennas_contrib_score[i] = 0;
        //antennas[i].x = rand() % W;
        //antennas[i].y = rand() % H;
    }
    file.close();

    optimisticScore = (long int *)malloc(M * sizeof(long int));
    FILE *foptim = fopen("antennas_contrib_B.csv", "r");
    for (size_t a = 0; a < M; a++)
    {
        fscanf(foptim, "%ld,", optimisticScore + a);
    }
    fclose(foptim);
    qsort(antennas, M, sizeof(Antenna), compareAntennas);
    free(optimisticScore);

    cudaMemcpy(dev_buildings_map, buildings_map, H * W * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_buildings, buildings, N * sizeof(Building), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_antennas, antennas, M * sizeof(Antenna), cudaMemcpyHostToDevice);

    const dim3 threadsPerBlock(8, 8);
    const dim3 numBlocks(W / threadsPerBlock.x, H / threadsPerBlock.y);

    long int best_score = LONG_MIN;
    Antenna *best_solution = (Antenna *)malloc(M * sizeof(Antenna));

    for (size_t iteration = 0; iteration < 5; iteration++)
    {
        printf("Iteration\n");
        printf("Score: %.2f => %ld\n", best_score / 2078043619.0, best_score); //B
        // printf("Score: %.2f => %ld\n", best_score / 5247238794.0, best_score); //D
        // printf("Score: %.2f => %ld\n", best_score / 8109310667.0, best_score); //E
        // printf("Score: %.2f => %ld\n", best_score / 24908802532.0, best_score); //F
        for (size_t a_idx = 0; a_idx < M; a_idx++)
        {
            cudaMemset(dev_antennas_contrib_score, 0, M * sizeof(int));
            solutionEval<<<N / 1000, 1000>>>(dev_buildings, dev_antennas, M, dev_antennas_contrib_score);
            cudaMemcpy(antennas_contrib_score, dev_antennas_contrib_score, M * sizeof(int), cudaMemcpyDeviceToHost);

            long int score = 0;
            for (size_t a = 0; a < M; a++)
            {
                score += antennas_contrib_score[a];
            }

            if (score >= best_score)
            {
                best_score = score;
                cudaMemcpy(best_solution, antennas, M * sizeof(Antenna), cudaMemcpyHostToHost);
                memcmp(best_solution, antennas, M * sizeof(Antenna));
            }

            //Find best position for antenna a_idx
            antennaPositionEval<<<numBlocks, threadsPerBlock>>>(a_idx, antennas[a_idx], H, W, N, dev_buildings, dev_buildings_map, dev_score_map);
            cudaMemcpy(score_map, dev_score_map, H * W * sizeof(int), cudaMemcpyDeviceToHost);
            int pos_x = -1;
            int pos_y = -1;
            int best = 0;
            for (int x = 0; x < W; x++)
            {
                for (int y = 0; y < H; y++)
                {
                    if ((score_map[W * y + x] > best) | (pos_x == -1))
                    {
                        best = score_map[W * y + x];
                        pos_x = x;
                        pos_y = y;
                    }
                }
            }

            if (best > antennas_contrib_score[a_idx])
            {
                //Move antenna
                if ((antennas[a_idx].x != pos_x) | (antennas[a_idx].y != pos_y))
                {
                    //antenna_moved[a_idx] = 1;
                }
                antennas[a_idx].x = pos_x;
                antennas[a_idx].y = pos_y;
                antennas[a_idx].placed = 1;
            }
            else if (best == antennas_contrib_score[a_idx])
            {
                antennas[a_idx].placed = 1;
            }
            else
            {
                //Unplace antenna
                antennas[a_idx].placed = 0;
            }

            cudaMemcpy(dev_antennas, antennas, M * sizeof(Antenna), cudaMemcpyHostToDevice);
        }
    }

    //Recompute best solution
    cudaMemset(dev_antennas_contrib_score, 0, M * sizeof(int));
    cudaMemcpy(dev_antennas, best_solution, M * sizeof(Antenna), cudaMemcpyHostToDevice);
    solutionEval<<<N / 1000, 1000>>>(dev_buildings, dev_antennas, M, dev_antennas_contrib_score);
    cudaMemcpy(antennas_contrib_score, dev_antennas_contrib_score, M * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(buildings, dev_buildings, M * sizeof(int), cudaMemcpyDeviceToHost);
    long int score = 0;
    int unused_antenna = 0;
    int *antennas_map = (int *)calloc(H * W, sizeof(int));
    int total_conflicts = 0;
    for (size_t a = 0; a < M; a++)
    {
        score += antennas_contrib_score[a];
        if (best_solution[a].placed == 0)
        {
            unused_antenna++;
        }
        int n = antennas_map[W * best_solution[a].y + best_solution[a].x];
        if (n > 0)
        {
            total_conflicts++;
        }
        antennas_map[W * best_solution[a].y + best_solution[a].x]++;
    }
    free(antennas_map);
    int not_connected_buildings = 0;
    for (size_t b = 0; b < N; b++)
    {
        if (buildings[b].antenna_connected_to == -1)
        {
            not_connected_buildings++;
        }
    }

    printf("Best score achieved: %ld\n", best_score);
    printf("Number of antennas not used %d/%d\n", unused_antenna, M);
    printf("Number of buildings not connected %d/%d\n", not_connected_buildings, N);
    printf("Number of conflicts %d\n", total_conflicts);

    // printf("%d\n",M);
    // for (size_t a = 0; a < M; a++)
    // {
    //     printf("%d %d %d\n",best_solution[a].ID,best_solution[a].x,best_solution[a].y);
    // }

    free(best_solution);

    cudaFreeHost(buildings);
    cudaFreeHost(antennas);
    cudaFreeHost(antennas_contrib_score);
    cudaFree(dev_buildings);
    cudaFree(dev_antennas);
    cudaFree(dev_antennas_contrib_score);

    free(buildings_map);
    free(score_map);
    cudaFree(dev_buildings_map);
    cudaFree(dev_score_map);

    return 0;
}
