#include <stdio.h>
#include <stdlib.h>
#define POPULATION 64
#define ELITE 8
#define BOT 16
#define RHO 70

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

struct Solution
{
    int *antennas_position;
    int *antennas_score;
    long int score;
};

int cmpfunc(const void *a, const void *b)
{
    return -(((Solution *)a)->score - ((Solution *)b)->score);
}

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

__global__ void solutionEval(const Problem dev_P, const int *antennas_positions, int *building_score, int *building_antenna)
{
    const int build_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_contribution = 0;
    int connected_antenna = -1;
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
                if ((contrib > max_contribution) | (connected_antenna == -1))
                {
                    max_contribution = contrib;
                    connected_antenna = 1;
                }
            }
        }
    }
    building_score[build_idx] = max_contribution;
    building_antenna[build_idx] = connected_antenna;
    //printf("Chiamata GPU per building %d\nMax contribution %d\n Best antenna %d\n\n", build_idx, max_contribution, antenna_idx);
}

__global__ void getAntennasScore(const Problem dev_P, const int *antennas_positions, const int *building_antenna, int *antennas_score)
{
    const int ant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = antennas_positions[2 * ant_idx];
    const int c = antennas_positions[2 * ant_idx + 1];
    const int s = dev_P.antennas_speed[ant_idx];
    //const int R = dev_P.antennas_range[ant_idx];
    long int contrib = 0;
    for (size_t b = 0; b < dev_P.N; b++)
    {
        if (building_antenna[b] == ant_idx)
        {
            const int dist = abs(dev_P.buildings_c[b] - c) + abs(dev_P.buildings_r[b] - r);
            contrib += dev_P.buildings_connection_speed[b] * s - dev_P.buildings_latency[b] * dist;
        }
    }
    antennas_score[ant_idx] = contrib;
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
    int *building_score;
    int *dev_antennas_positions, *dev_building_score, *dev_building_antenna, *dev_antennas_score;
    cudaMalloc((void **)&dev_antennas_positions, 2 * P.M * sizeof(int));
    cudaHostAlloc((void **)&building_score, P.N * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&dev_building_score, P.N * sizeof(int));
    cudaMalloc((void **)&dev_building_antenna, P.N * sizeof(int));
    cudaMalloc((void **)&dev_antennas_score, P.M * sizeof(int));

    srand(time(NULL));

    Solution *population = (Solution *)malloc(POPULATION * sizeof(Solution));
    for (size_t p = 0; p < POPULATION; p++)
    {
        population[p].antennas_position = (int *)malloc(2 * P.M * sizeof(int));
        population[p].antennas_score = (int *)malloc(P.M * sizeof(int));
        for (size_t a = 0; a < P.M; a++)
        {
            population[p].antennas_position[2 * a] = rand() % P.W;
            population[p].antennas_position[2 * a + 1] = rand() % P.H;
        }
    }

    //Insert good starting pos
    // FILE * goodstartpos = fopen("../good_starting_pos/b.txt","r");
    // fscanf(goodstartpos,"%*d\n");
    // for (size_t a = 0; a < P.M; a++)
    // {
    //     int a_idx, x,y;
    //     fscanf(goodstartpos,"%d %d %d", &a_idx, &x, &y);
    //     population[0].antennas_position[2*a_idx] = x;
    //     population[0].antennas_position[2*a_idx + 1] = y;
    // }
    // fclose(goodstartpos);

    for (size_t epoch = 0; epoch < 1000; epoch++)
    {
        for (size_t p = 0; p < POPULATION; p++)
        {
            //Copy a solution/individual to GPU
            cudaMemcpy(dev_antennas_positions, population[p].antennas_position, 2 * P.M * sizeof(int), cudaMemcpyHostToDevice);
            //Call GPU kernels
            solutionEval<<<P.N / 1000, 1000>>>(dev_P, dev_antennas_positions, dev_building_score, dev_building_antenna);
            getAntennasScore<<<P.M / 1014, 1014>>>(dev_P, dev_antennas_positions, dev_building_antenna, dev_antennas_score);

            //Copy results back
            cudaMemcpy(population[p].antennas_score, dev_antennas_score, P.M * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(building_score, dev_building_score, P.N * sizeof(int), cudaMemcpyDeviceToHost);
            //Evaluate results
            long int score = 0;
            for (size_t b = 0; b < P.N; b++)
            {
                score += building_score[b];
            }
            population[p].score = score;
        }

        //Sort population according to score
        qsort(population, POPULATION, sizeof(Solution), cmpfunc);
        printf("Epoch %lu. Best solution score: %ld\n", epoch, population[0].score);

        //Crossover
        for (size_t p = ELITE; p < POPULATION - BOT; p++)
        {
            int elite_parent = rand() % ELITE;
            int nonelite_parent = ELITE + (rand() % (POPULATION - ELITE));
            for (size_t a = 0; a < P.M; a++)
            {
                if ((population[nonelite_parent].antennas_score[a] > population[elite_parent].antennas_score[a]) | (rand() % 100 > RHO))
                {
                    population[p].antennas_position[2 * a] = population[nonelite_parent].antennas_position[2 * a];
                    population[p].antennas_position[2 * a + 1] = population[nonelite_parent].antennas_position[2 * a + 1];
                }
                else
                {
                    population[p].antennas_position[2 * a] = population[elite_parent].antennas_position[2 * a];
                    population[p].antennas_position[2 * a + 1] = population[elite_parent].antennas_position[2 * a + 1];
                }
            }
        }

        //Randomly generate mutant solutions
        for (size_t p = POPULATION - BOT; p < POPULATION; p++)
        {
            for (size_t a = 0; a < P.M; a++)
            {
                population[p].antennas_position[2 * a] = rand() % P.W;
                population[p].antennas_position[2 * a + 1] = rand() % P.H;
            }
        }
    }

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

    for (size_t p = 0; p < POPULATION; p++)
    {
        free(population[p].antennas_position);
    }
    free(population);

    cudaFreeHost(building_score);
    return 0;
}
