// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

#include <iostream>
#include <fstream>
#include <stdio.h>

struct Antenna
{
    int range, speed, ID; //const?
    int x, y;
};

// int compareAntennas (const void * a, const void * b) {
//    return ( ((Antenna*)b)->range - ((Antenna*)a)->range); //Sort by decreasing range
// }

struct Building
{
    int x, y, latency, speed; //const?
    int antenna_connected_to;
    int best_score;
};

__global__ void solutionEval(Building *buildings, const Antenna *antennas, const int M)
{
    const int build_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const Building b = buildings[build_idx];
    long int best_score = 0;
    int antenna_connected_to = -1;
    for (size_t a_idx = 0; a_idx < M; a_idx++)
    {
        const Antenna a = antennas[a_idx];
        const int dist = abs(b.x - a.x) + abs(b.y - a.y);
        if (dist <= a.range)
        {
            const int contrib = b.speed * a.speed - b.latency * dist;
            if ((contrib > best_score) | (antenna_connected_to == -1))
            {
                best_score = contrib;
                antenna_connected_to = a_idx;
            }
        }
    }
    buildings[build_idx].best_score = best_score;
    buildings[build_idx].antenna_connected_to = antenna_connected_to;
}

__global__ void antennaPositionEval(const Antenna a, const int H, const int W, const Building *buildings, const int *dev_buildings_map, int *score_map)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    //printf("(%d,%d)\n",y,x);

    int score = 0;
    for (int yy = max(0, y - a.range); yy <= min(H - 1, y + a.range); yy++)
    {
        const int dif = a.range - abs(y - yy);
        for (int xx = max(0, x - dif); xx <= min(W - 1, x + dif); xx++)
        {
            const int build_idx = dev_buildings_map[W * yy + xx];
            if (build_idx != -1)
            {
                Building b = buildings[build_idx];
                const int dist = abs(y - yy) + abs(x - xx);
                const int contrib = b.speed * a.speed - b.latency * dist;
                if (contrib > b.best_score)
                {
                    score += (contrib - b.best_score);
                }
                else if (b.antenna_connected_to == -1)
                {
                    score += contrib;
                }
            }
        }
    }
    //printf("called host with (%d,%d): %d\n",r,c,score);
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

    int *buildings_map, *score_map;
    int *dev_buildings_map, *dev_score_map;

    int W, H, N, M, R;
    std::ifstream file(argv[1], std::ios::in);
    file >> W >> H >> N >> M >> R;

    cudaHostAlloc((void **)&buildings, N * sizeof(Building), cudaHostAllocDefault);
    cudaHostAlloc((void **)&antennas, M * sizeof(Antenna), cudaHostAllocDefault);
    cudaMalloc((void **)&(dev_buildings), N * sizeof(Building));
    cudaMalloc((void **)&(dev_antennas), M * sizeof(Antenna));

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
        antennas[i].x = rand() % W;
        antennas[i].y = rand() % H;
    }
    file.close();

    //qsort(antennas, M, sizeof(Antenna),compareAntennas);

    cudaMemcpy(dev_buildings_map, buildings_map, H * W * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_buildings, buildings, N * sizeof(Building), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_antennas, antennas, M * sizeof(Antenna), cudaMemcpyHostToDevice);

    const dim3 threadsPerBlock(8, 8);
    const dim3 numBlocks(W / threadsPerBlock.x, H / threadsPerBlock.y);

    for (size_t iteration = 0; iteration < 10000; iteration++)
    {
        solutionEval<<<N / 1000, 1000>>>(dev_buildings, dev_antennas, M);

        if (iteration % 100 == 0)
        {
            cudaMemcpy(buildings, dev_buildings, N * sizeof(Building), cudaMemcpyDeviceToHost);
            long int score = 0;
            for (int b = 0; b < N; b++)
            {
                if (buildings[b].antenna_connected_to != -1)
                {
                    score += buildings[b].best_score;
                }
            }
            std::cout << (score * 1.0) / (2078043619.0) << std::endl; //B
            //std::cout << (score * 1.0) / (5247238794.0) << std::endl; //D
            //std::cout << (score * 1.0) / (8109310667.0) << std::endl; //F
        }

        //Find best position for antenna
        int a_idx = rand() % M;
        antennaPositionEval<<<numBlocks, threadsPerBlock>>>(antennas[a_idx], H, W, dev_buildings, dev_buildings_map, dev_score_map);
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

        //Move antenna
        antennas[a_idx].x = pos_x;
        antennas[a_idx].y = pos_y;
        cudaMemcpy(dev_antennas, antennas, M * sizeof(Antenna), cudaMemcpyHostToDevice);
    }

    // for (size_t a = 0; a < M; a++)
    // {
    //     std::cout << antennas[a].ID << ": (" << antennas[a].x << "," << antennas[a].y << ")\n";
    // }

    cudaFreeHost(buildings);
    cudaFreeHost(antennas);
    cudaFree(dev_buildings);
    cudaFree(dev_antennas);

    free(buildings_map);
    free(score_map);
    cudaFree(dev_buildings_map);
    cudaFree(dev_score_map);

    return 0;
}
