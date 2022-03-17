#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define NUM_ACTIONS (2)

typedef struct
{
    unsigned int stamina, turn;
} state;

int main(int argc, char const *argv[])
{
    Problem prob = parse_input_file(argv[1]);

    // unsigned int ***Q_values_table = (unsigned int ***)malloc(sizeof(unsigned int ***) * NUM_ACTIONS);
    // for (size_t a = 0; a < NUM_ACTIONS; a++)
    // {
    //     Q_values_table[a] = (unsigned int **)malloc(sizeof(unsigned int **) * prob.num_turns);
    //     for (size_t t = 0; t < prob.num_turns; t++)
    //     {
    //         Q_values_table[a][t] = (unsigned int *)calloc(prob.stamina_max, sizeof(unsigned int *));
    //     }
    // }

    int list[] = {1,3,2,0,4};

    long unsigned int reward = complete_simulator(prob.num_demons, list, &prob);

    printf("Input file %s. Reward %ld\n", argv[1], reward);

    return 0;
}
