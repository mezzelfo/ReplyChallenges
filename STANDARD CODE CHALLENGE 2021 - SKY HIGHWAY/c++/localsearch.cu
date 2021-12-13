#include <stdio.h>
#include <stdlib.h>

#define SIGN(x) ((0 < (x)) - ((x) < 0))

struct Problem
{
    int H, W, N, M, R;
    int *buildings_c;
    int *buildings_r;
    int *buildings_latency;
    int *buildings_connection_speed;
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
    P.buildings_c = (int *)calloc(P.N, sizeof(int));
    P.buildings_r = (int *)calloc(P.N, sizeof(int));
    P.buildings_latency = (int *)calloc(P.N, sizeof(int));
    P.buildings_connection_speed = (int *)calloc(P.N, sizeof(int));
    P.antennas_range = (int *)malloc(P.M * sizeof(int));
    P.antennas_speed = (int *)malloc(P.M * sizeof(int));

    if ((P.buildings_c == NULL) || (P.buildings_r == NULL) || (P.buildings_latency == NULL) || (P.buildings_connection_speed == NULL) || (P.antennas_range == NULL) || (P.antennas_speed == NULL))
    {
        printf("ERROR: Unable to malloc\n");
        exit(-2);
    }

    // Filling data structures
    for (int i = 0; i < P.N; i++)
    {
        int c, r, l, s;
        fscanf(fileptr, "%d %d %d %d\n", &c, &r, &l, &s);
        P.buildings_c[i] = c;
        P.buildings_r[i] = r;
        P.buildings_latency[i] = l;
        P.buildings_connection_speed[i] = s;
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

__global__ void improveSolution(const Problem dev_P, const int *building_score, const int *building_antenna, int *antennas_positions)
{
    const int ant_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int used = 0;

    const int r = antennas_positions[2 * ant_idx];
    const int c = antennas_positions[2 * ant_idx + 1];
    //const int ant_range = dev_P.antennas_range[ant_idx];
    //const int ant_speed = dev_P.antennas_speed[ant_idx];

    long int total_r = 0;
    long int total_c = 0;
    for (size_t b = 0; b < dev_P.N; b++)
    {
        if (building_antenna[b] == ant_idx)
        {
            used = 1;
            const long int multiplier = building_score[b];
            const int dr = r - dev_P.buildings_r[b];
            const int dc = c - dev_P.buildings_c[b];
            total_r += SIGN(dr) * multiplier;
            total_c += SIGN(dc) * multiplier;
        }
    }

    antennas_positions[2 * ant_idx] = r - SIGN(total_r);
    antennas_positions[2 * ant_idx + 1] = c - SIGN(total_c);

    if (!used)
    {
        antennas_positions[2 * ant_idx] = (257*r+139) % dev_P.W;
        antennas_positions[2 * ant_idx + 1] = (181*c+37) % dev_P.H;
    }
}

__global__ void solutionEval(const Problem dev_P, const int *antennas_positions, int *building_score, int *building_antenna)
{
    const int build_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int antenna_idx = -1;
    int max_contribution = 0;
    for (size_t a = 0; a < dev_P.M; a++)
    {
        //Naive implementation
        // const int dist = abs(dev_P.buildings_c[build_idx] - antennas_positions[2 * a]) + abs(dev_P.buildings_r[build_idx] - antennas_positions[2 * a + 1]);
        // if (dist <= dev_P.antennas_range[a])
        // {
        //     const int long contrib = dev_P.buildings_connection_speed[build_idx] * dev_P.antennas_speed[a] - dev_P.buildings_latency[build_idx] * dist;
        //     if (contrib > max_contribution)
        //     {
        //         max_contribution = contrib;
        //         antenna_idx = a;
        //     }
        // }
        //Suddivido in due il calcolo di dist per efficienza
        int dist = abs(dev_P.buildings_c[build_idx] - antennas_positions[2 * a]);
        if (dist <= dev_P.antennas_range[a])
        {
            dist += abs(dev_P.buildings_r[build_idx] - antennas_positions[2 * a + 1]);
            if (dist <= dev_P.antennas_range[a])
            {
                int contrib = dev_P.buildings_connection_speed[build_idx] * dev_P.antennas_speed[a] - dev_P.buildings_latency[build_idx] * dist;
                if ((contrib > max_contribution) | (antenna_idx == -1))
                {
                    max_contribution = contrib;
                    antenna_idx = a;
                }
            }
        }
    }
    building_score[build_idx] = max_contribution;
    building_antenna[build_idx] = antenna_idx;
    //printf("Chiamata GPU per building %d\nMax contribution %d\n Best antenna %d\n\n", build_idx, max_contribution, antenna_idx);
}

long int computeScore(Problem P, int *building_score, int *building_antenna)
{
    //Compute score
    long int score = 0;
    bool reward = 1;
    for (size_t b = 0; b < P.N; b++)
    {
        score += building_score[b];
        reward &= (building_antenna[b] != -1);
    }
    if (reward)
    {
        score += P.R;
    }
    return score;
}

int main(int argc, char const *argv[])
{
    printf("Inizio lettura file\n");
    Problem P = read_file(argv[1]);
    printf("File letto correttamente\n");

    // Copy only W H N M R
    Problem dev_P;
    dev_P.W = P.W;
    dev_P.H = P.H;
    dev_P.N = P.N;
    dev_P.M = P.M;
    dev_P.R = P.R;

    cudaMalloc((void **)&(dev_P.buildings_c), P.N * sizeof(int));
    cudaMalloc((void **)&(dev_P.buildings_r), P.N * sizeof(int));
    cudaMalloc((void **)&(dev_P.buildings_latency), P.N * sizeof(int));
    cudaMalloc((void **)&(dev_P.buildings_connection_speed), P.N * sizeof(int));
    cudaMalloc((void **)&(dev_P.antennas_range), P.M * sizeof(int));
    cudaMalloc((void **)&(dev_P.antennas_speed), P.M * sizeof(int));

    cudaMemcpy(dev_P.buildings_c, P.buildings_c, P.N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_P.buildings_r, P.buildings_r, P.N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_P.buildings_latency, P.buildings_latency, P.N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_P.buildings_connection_speed, P.buildings_connection_speed, P.N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_P.antennas_range, P.antennas_range, P.M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_P.antennas_speed, P.antennas_speed, P.M * sizeof(int), cudaMemcpyHostToDevice);

    // Output buffers
    int *antennas_positions, *building_score, *building_antenna;
    int *dev_antennas_positions, *dev_building_score, *dev_building_antenna;
    cudaHostAlloc((void **)&antennas_positions, 2 * P.M * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&dev_antennas_positions, 2 * P.M * sizeof(int));
    cudaHostAlloc((void **)&building_score, P.N * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&dev_building_score, P.N * sizeof(int));
    cudaHostAlloc((void **)&building_antenna, P.N * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&dev_building_antenna, P.N * sizeof(int));

    srand(time(NULL));

    //Assign solution
    for (size_t a = 0; a < P.M; a++)
    {
        antennas_positions[2 * a] = rand() % P.W;
        antennas_positions[2 * a + 1] = rand() % P.H;
    }

    //Insert good starting pos
    FILE *goodstartpos = fopen("../good_starting_pos/b.txt", "r");
    fscanf(goodstartpos, "%*d\n");
    for (size_t a = 0; a < P.M; a++)
    {
        int a_idx, x, y;
        fscanf(goodstartpos, "%d %d %d", &a_idx, &x, &y);
        antennas_positions[2 * a_idx] = x;
        antennas_positions[2 * a_idx + 1] = y;
    }
    fclose(goodstartpos);

    //Copy solution to GPU
    cudaMemcpy(dev_antennas_positions, antennas_positions, 2 * P.M * sizeof(int), cudaMemcpyHostToDevice);

    //Call GPU for each building
    solutionEval<<<P.N / 1000, 1000>>>(dev_P, dev_antennas_positions, dev_building_score, dev_building_antenna);

    //Copy results back
    cudaMemcpy(building_score, dev_building_score, P.N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(building_antenna, dev_building_antenna, P.N * sizeof(int), cudaMemcpyDeviceToHost);
    //Print Score
    printf("Score %ld\n", computeScore(P, building_score, building_antenna));

    for (size_t times = 0; times < 1; times++)
    {
        //Call GPU for each antenna
        improveSolution<<<P.M / 1014, 1014>>>(dev_P, dev_building_score, dev_building_antenna, dev_antennas_positions);

        //Call GPU for each building
        solutionEval<<<P.N / 1000, 1000>>>(dev_P, dev_antennas_positions, dev_building_score, dev_building_antenna);

        //Copy results back
        cudaMemcpy(building_score, dev_building_score, P.N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(building_antenna, dev_building_antenna, P.N * sizeof(int), cudaMemcpyDeviceToHost);
        //Print Score
        printf("Score %ld\n", computeScore(P, building_score, building_antenna));
    }

    // int *final_antennas_positions;
    // cudaHostAlloc((void **)&final_antennas_positions, 2 * P.M * sizeof(int), cudaHostAllocDefault);
    // cudaMemcpy(final_antennas_positions, dev_antennas_positions, 2 * P.M * sizeof(int), cudaMemcpyDeviceToHost);

    // for (size_t a = 0; a < P.M; a++)
    // {
    //     int dc = final_antennas_positions[2*a]-antennas_positions[2*a];
    //     int dy = final_antennas_positions[2*a+1]-antennas_positions[2*a+1];
    //     printf("dc %d, dy %d\n", dc, dy);
    // }

    cudaFree(dev_P.buildings_c);
    cudaFree(dev_P.buildings_r);
    cudaFree(dev_P.buildings_latency);
    cudaFree(dev_P.buildings_connection_speed);
    cudaFree(dev_P.antennas_range);
    cudaFree(dev_P.antennas_speed);

    free(P.buildings_c);
    free(P.buildings_r);
    free(P.buildings_latency);
    free(P.buildings_connection_speed);
    free(P.antennas_range);
    free(P.antennas_speed);

    cudaFreeHost(antennas_positions);
    cudaFreeHost(building_score);
    cudaFreeHost(building_antenna);

    return 0;
}
