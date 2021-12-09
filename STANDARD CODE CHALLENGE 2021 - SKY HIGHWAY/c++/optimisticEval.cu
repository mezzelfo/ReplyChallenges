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

__global__ void optimisticEval(const Problem dev_P, const int ant_range, const int ant_speed, int *optimistic_score_map)
{
    const int r = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int c = (blockIdx.y * blockDim.y) + threadIdx.y;

    int sum = 0;
    for (int rr = max(0, r - ant_range); rr <= min(dev_P.H - 1, r + ant_range); rr++)
    {
        const int dif = ant_range - abs(r - rr);
        for (int cc = max(0, c - dif); cc <= min(dev_P.W - 1, c + dif); cc++)
        {
            const int dist = abs(r - rr) + abs(c - cc);
            const int long contrib = dev_P.connection_speed_map[dev_P.W * rr + cc] * ant_speed - dev_P.latency_map[dev_P.W * rr + cc] * dist;
            if (contrib > 0)
            {
                sum += contrib;
            }
        }
    }
    optimistic_score_map[dev_P.W * r + c] = sum;
}


int main(int argc, char const *argv[])
{
    printf("Inizio lettura file\n");
    Problem P = read_file(argv[1]);
    printf("File letto correttamente\n");

    // Copy only W H N M R
    Problem dev_P = P;
    dev_P.latency_map = NULL;
    dev_P.connection_speed_map = NULL;
    dev_P.antennas_range = NULL;
    dev_P.antennas_speed = NULL;

    // Output buffers
    int *optimistic_score_map, *dev_optimistic_score_map;
    cudaHostAlloc((void **)&optimistic_score_map, P.W * P.H * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&dev_optimistic_score_map, P.W * P.H * sizeof(int));

    cudaMalloc((void **)&(dev_P.latency_map), P.W * P.H * sizeof(int));
    cudaMalloc((void **)&(dev_P.connection_speed_map), P.W * P.H * sizeof(int));
    cudaMemcpy(dev_P.latency_map, P.latency_map, P.W * P.H * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_P.connection_speed_map, P.connection_speed_map, P.W * P.H * sizeof(int), cudaMemcpyHostToDevice);

    int *antennas_pos = (int*)malloc(P.M * sizeof(int));

    const dim3 threadsPerBlock(8, 8);
    const dim3 numBlocks(P.W / threadsPerBlock.x, P.H / threadsPerBlock.y);
    long int total = 0;
    for (int a = 0; a < P.M; a++)
    {
        //printf("%d/%d\n", a, P.M);
        const int ant_range = P.antennas_range[a];
        const int ant_speed = P.antennas_speed[a];
        optimisticEval<<<numBlocks, threadsPerBlock>>>(dev_P, ant_range, ant_speed, dev_optimistic_score_map);
        cudaMemcpy(optimistic_score_map, dev_optimistic_score_map, P.W * P.H * sizeof(int), cudaMemcpyDeviceToHost);
        int max_val = 0;
        for (size_t i = 0; i < P.H * P.W; i++)
        {
            if (optimistic_score_map[i] > max_val)
            {
                max_val = optimistic_score_map[i];
                antennas_pos[a] = i;
            }
        }
        total += max_val;
    }

    printf("total: %ld\n",total);

    FILE* fout = fopen("fout.csv","w");
    for (int a = 0; a < P.M; a++)
    {
        fprintf(fout,"%d,",antennas_pos[a]);
    }
    fclose(fout);

    FILE* fmapout = fopen("fmapout.csv","w");
    for (size_t i = 0; i < P.H * P.W; i++)
    {
        fprintf(fmapout,"%d,",optimistic_score_map[i]);
    }
    fclose(fmapout);

    free(P.latency_map);
    free(P.connection_speed_map);
    free(P.antennas_range);
    free(P.antennas_speed);

    cudaFree(dev_P.latency_map);
    cudaFree(dev_P.connection_speed_map);

    free(antennas_pos);

    cudaFreeHost(optimistic_score_map);
    
    return 0;
}
