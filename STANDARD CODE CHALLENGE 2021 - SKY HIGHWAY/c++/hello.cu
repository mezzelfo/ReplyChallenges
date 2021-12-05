#include <stdio.h>

struct Problem
{
    int H, W, N, M, R;
    int *latency_map;
    int *connection_speed_map;
    int *antennas_range;
    int *antennas_speed;
};

Problem read_file(const char *filename)
{
    FILE *fileptr = fopen(filename, "r");
    if (!fileptr)
    {
        printf("ERROR: File %s not found\n", filename);
        exit(-1);
    }

    Problem P;

    fscanf(fileptr, "%d %d %d %d %d", &(P.W), &(P.H), &(P.N), &(P.M), &(P.R));

    //Allocation
    P.latency_map = (int *)calloc(P.W * P.H, sizeof(int));
    P.connection_speed_map = (int *)calloc(P.W * P.H, sizeof(int));
    P.antennas_range = (int *)malloc(P.M * sizeof(int));
    P.antennas_speed = (int *)malloc(P.M * sizeof(int));

    if ((P.latency_map == NULL) || (P.connection_speed_map == NULL) || (P.antennas_range == NULL) || (P.antennas_speed == NULL))
    {
        printf("ERROR: Unable to malloc\n");
        exit(-2);
    }

    // Filling data structures
    for (int i = 0; i < P.N; i++)
    {
        int c, r, l, s;
        fscanf(fileptr, "%d %d %d %d\n", &c, &r, &l, &s);
        P.latency_map[P.W * r + c] = l;
        P.connection_speed_map[P.W * r + c] = s;
    }
    for (int i = 0; i < P.M; i++)
    {
        int r, s;
        fscanf(fileptr, "%d %d\n", &r, &s);
        P.antennas_range[i] = r;
        P.antennas_speed[i] = s;
    }
    fclose(fileptr);

    return P;
}

__global__ void optimisticEval(const int *latency_map, const int *connection_speed_map, int *optimistic_score_map, const int ant_range, const int ant_speed, const int W, const int H)
{
    const int r = blockIdx.x;
    const int c = threadIdx.x;
    int sum = 0;
    for (int rr = max(0, r - ant_range); rr <= min(H - 1, r + ant_range); rr++)
    {
        const int dif = ant_range - abs(r - rr);
        for (int cc = max(0, c - dif); cc <= min(W - 1, c + dif); cc++)
        {
            const int dist = abs(r - rr) + abs(c - cc);
            const int long contrib = connection_speed_map[blockDim.x * rr + cc] * ant_speed - latency_map[blockDim.x * rr + cc] * dist;
            if (contrib > 0)
            {
                sum += contrib;
            }
        }
    }
    optimistic_score_map[blockDim.x * r + c] = sum;
}

int main(int argc, char const *argv[])
{
    printf("Inizio lettura file\n");
    Problem P = read_file(argv[1]);
    printf("File letto correttamente\n");

    int *optimistic_score_map = (int*)malloc(P.W * P.H * sizeof(int));

    int *dev_latency_map, *dev_connection_speed_map, *dev_optimistic_score_map;
    cudaMalloc((void **)&(dev_latency_map), P.W * P.H * sizeof(int));
    cudaMalloc((void **)&dev_connection_speed_map, P.W * P.H * sizeof(int));
    cudaMalloc((void **)&dev_optimistic_score_map, P.W * P.H * sizeof(int));
    cudaMemcpy(dev_latency_map, P.latency_map, P.W * P.H * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_connection_speed_map, P.connection_speed_map, P.W * P.H * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_optimistic_score_map, optimistic_score_map, P.W * P.H * sizeof(int), cudaMemcpyHostToDevice);

    
    long optimistic_score = 0;
    for (int a = 0; a < P.M; a++)
    {
        int ant_range = P.antennas_range[a];
        int ant_speed = P.antennas_speed[a];
        optimisticEval<<<P.H, P.W>>>(dev_latency_map, dev_connection_speed_map, dev_optimistic_score_map, ant_range, ant_speed, P.W, P.H);
        cudaMemcpy(optimistic_score_map, dev_optimistic_score_map, P.W * P.H * sizeof(int), cudaMemcpyDeviceToHost);

        long int contrib = 0;
        for (int r = 0; r < P.H; r++)
        {
            for (int c = 0; c < P.W; c++)
            {
                if (optimistic_score_map[P.W * r + c] > contrib)
                {
                    contrib = optimistic_score_map[P.W * r + c];
                }
            }
        }
        optimistic_score += contrib;
    }
    printf("optimistic_score: %ld ", optimistic_score);
    return 0;
}