#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <tuple>
#include <algorithm>

struct Problem
{
    int H, W, N, M, R;
    int *buildings_c;
    int *buildings_r;
    int *buildings_latency;
    int *buildings_connection_speed;
    int *antennas_range;
    int *antennas_speed;
    std::vector<std::tuple<int, int, int>> antennas;
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
        P.antennas.emplace_back(0,r,s);
    }
    fclose(fileptr);

    return P;
}

std::vector<std::pair<int, int>> antennas_positions;
std::vector<std::pair<int, int>> best_solution;
std::vector<int> optimisticContrib;
long int best_score = 0;
Problem P;
long int target_score;

bool valid_position(const std::pair<int, int> &pos, const int current_antenna)
{
    for (size_t a = 0; a < current_antenna; a++)
    {
        if (antennas_positions[a] == pos)
        {
            return false;
        }
    }
    return true;
}

long int solutionEval(int current_antenna)
{
    long int score = 0;
    bool reward = true;
    for (size_t b = 0; b < P.N; b++)
    {
        long int max_contribution = 0;
        int antenna_idx = -1;
        for (size_t a = 0; a < current_antenna; a++)
        {
            const int dist = abs(P.buildings_c[b] - antennas_positions[a].first) + abs(P.buildings_r[b] - antennas_positions[a].second);
            if (dist <= P.antennas_range[a])
            {
                const int long contrib = P.buildings_connection_speed[b] * P.antennas_speed[a] - P.buildings_latency[b] * dist;
                if ((contrib > max_contribution) | (antenna_idx == -1))
                {
                    max_contribution = contrib;
                    antenna_idx = a;
                }
            }
        }
        score += max_contribution;
        if (antenna_idx == -1)
        {
            reward = false;
        }
    }
    if (reward)
    {
        score += P.R;
    }

    if (score > best_score)
    {
        best_score = score;
        best_solution.assign(antennas_positions.begin(), antennas_positions.end());
        printf("New best score %ld. current antenna %d\n", best_score, current_antenna);
    }

    return score;
}

long int optimisticEvalFromNowOn(int current_antenna)
{
    long int sum = 0;
    for (size_t a = current_antenna; a < P.M; a++)
    {
        sum += optimisticContrib[a];
    }
    return sum;
}

void backtrack(int current_antenna)
{
    long int current_solution_score = solutionEval(current_antenna);
    if (current_antenna >= P.M)
    {
        return;
    }
    long int delta = optimisticEvalFromNowOn(current_antenna);
    if ((current_solution_score < 10000000) & (current_antenna > 0))
    {
        return;
    }
    if ((current_solution_score < 50000000) & (current_antenna > 5))
    {
        return;
    }
    
    if (current_solution_score + delta < target_score)
        return;
    for (size_t x = 0; x < P.W; x+=150) //Columns
    {
        for (size_t y = 0; y < P.H; y+=150) //Rows
        {
            const auto pos = std::make_pair<int, int>(x, y);
            if (valid_position(pos, current_antenna))
            {
                antennas_positions[current_antenna] = pos;
                backtrack(current_antenna + 1);
            }
        }
    }
}

bool sortdesc(const std::tuple<int, int, int>& a,
              const std::tuple<int, int, int>& b)
{
    return (std::get<0>(a) > std::get<0>(b));
}

int main(int argc, char const *argv[])
{
    printf("Inizio lettura file\n");
    P = read_file(argv[1]);
    printf("File letto correttamente\n");

    optimisticContrib.reserve(P.M);
    FILE *fin = fopen("antennas_contrib_B.csv", "r");
    for (size_t a = 0; a < P.M; a++)
    {
        int contrib;
        fscanf(fin, "%d,", &contrib);
        std::get<0>(P.antennas[a]) = contrib;
    }
    fclose(fin);
    std::sort(P.antennas.begin(), P.antennas.end(), sortdesc);
    for (size_t a = 0; a < P.M; a++)
    {
        P.antennas_range[a] = std::get<1>(P.antennas[a]);
        P.antennas_speed[a] = std::get<2>(P.antennas[a]);
        optimisticContrib.push_back(std::get<0>(P.antennas[a]));
    }
    P.antennas.clear();
    P.antennas.shrink_to_fit();
    
    optimisticEvalFromNowOn(10);

    const std::pair<int, int> null_pos = std::make_pair<int, int>(-1, -1);
    antennas_positions.assign(P.M, null_pos);
    best_solution.assign(P.M, null_pos);

    target_score = 2078043619;

    backtrack(0);

    // for (size_t a = 0; a < P.M; a++)
    // {
    //     printf("Antenna %lu: (%d,%d)\n", a, best_solution[a].first, best_solution[a].second);
    // }
    printf("Final score %ld\n", best_score);
    return 0;
}
