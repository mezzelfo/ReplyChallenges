#include <stdio.h>

#define W 1200
#define H 1200

void read_file(const char *filename, int *N, int *M, int *R, int latency_map[H][W], int connection_speed_map[H][W], int **antennas_range, int **antennas_speed)
{
    FILE *fileptr = fopen(filename, "r");
    if (!fileptr)
    {
        printf("ERROR: File %s not found\n", filename);
        exit(-1);
    }

    fscanf(fileptr, "%*d %*d %d %d %d", N, M, R);
    for (int i = 0; i < *N; i++)
    {
        int c, r, l, s;
        fscanf(fileptr, "%d %d %d %d\n", &c, &r, &l, &s);
        latency_map[r][c] = l;
        connection_speed_map[r][c] = s;
    }

    *antennas_range = (int *)malloc(*M * sizeof(int));
    *antennas_speed = (int *)malloc(*M * sizeof(int));

    if ((*antennas_range == NULL) || (*antennas_speed == NULL))
    {
        printf("ERROR: Unable to malloc\n");
        exit(-2);
    }
    

    for (int i = 0; i < *M; i++)
    {
        int r, s;
        fscanf(fileptr, "%d %d\n", &r, &s);
        (*antennas_range)[i] = r;
        (*antennas_speed)[i] = s;
    }
    fclose(fileptr);
}

__global__ void optimisticEval(const int *latency_map, const int *connection_speed_map, int *optimistic_score_map, const int ant_range, const int ant_speed)
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
    int latency_map[H][W] = {0};
    int connection_speed_map[H][W] = {0};
    int optimistic_score_map[H][W] = {0};
    int *antennas_range;
    int *antennas_speed;

    int N, M, R;
    printf("Inizio lettura file\n");
    read_file(argv[1], &N, &M, &R, latency_map, connection_speed_map, &antennas_range, &antennas_speed);
    printf("File letto correttamente\n");

    int *dev_latency_map, *dev_connection_speed_map, *dev_optimistic_score_map;
    cudaMalloc((void **)&dev_latency_map, W * H * sizeof(int));
    cudaMalloc((void **)&dev_connection_speed_map, W * H * sizeof(int));
    cudaMalloc((void **)&dev_optimistic_score_map, W * H * sizeof(int));
    cudaMemcpy(dev_latency_map, latency_map, W * H * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_connection_speed_map, connection_speed_map, W * H * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_optimistic_score_map, optimistic_score_map, W * H * sizeof(int), cudaMemcpyHostToDevice);

    long optimistic_score = 0;
    for (int a = 0; a < M; a++)
    {
        int ant_range = antennas_range[a];
        int ant_speed = antennas_speed[a];
        optimisticEval<<<H, W>>>(dev_latency_map, dev_connection_speed_map, dev_optimistic_score_map, ant_range, ant_speed);
        cudaMemcpy(optimistic_score_map, dev_optimistic_score_map, W * H * sizeof(int), cudaMemcpyDeviceToHost);

        long int contrib = 0;
        for (int r = 0; r < H; r++)
        {
            for (int c = 0; c < W; c++)
            {
                if (optimistic_score_map[r][c] > contrib)
                {
                    contrib = optimistic_score_map[r][c];
                }
            }
        }
        optimistic_score += contrib;
    }
    printf("optimistic_score: %ld ", optimistic_score);
    return 0;
}