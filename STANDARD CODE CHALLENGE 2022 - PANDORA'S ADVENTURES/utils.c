#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utils.h"

Problem parse_input_file(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    Problem P;
    int checkfscaf;
    checkfscaf = fscanf(fp, "%d %d %d %ld\n",
           &(P.stamina_init),
           &(P.stamina_max),
           &(P.num_turns),
           &(P.num_demons));
    
    assert(checkfscaf == 4);

    P.demons = (Demon *)malloc(sizeof(Demon) * P.num_demons);
    for (size_t i = 0; i < P.num_demons; i++)
    {
        checkfscaf = fscanf(fp, "%d %d %d %ld ",
               &(P.demons[i].stamina_consumed),
               &(P.demons[i].turn_recovered),
               &(P.demons[i].stamina_recovered),
               &(P.demons[i].num_turns_framgments));
        assert(checkfscaf == 4);
        P.demons[i].fragments = (unsigned int *)malloc(sizeof(unsigned int) * P.demons[i].num_turns_framgments);
        for (size_t j = 0; j < P.demons[i].num_turns_framgments; j++)
        {
            checkfscaf = fscanf(fp, "%d", &(P.demons[i].fragments[j]));
            assert(checkfscaf == 1);
        }
        checkfscaf = fscanf(fp, "\n");
    }

    return P;
}

long unsigned int complete_simulator(const int length_demon_idx_list, const int *demon_idx_list, const Problem *P)
{
    unsigned int stamina = P->stamina_init;
    unsigned int fragments = 0;
    size_t demon_pos_idx = 0;

    unsigned int *fragments_list = (unsigned int *)malloc(sizeof(unsigned int) * P->num_turns);
    unsigned int *stamina_list = (unsigned int *)malloc(sizeof(unsigned int) * P->num_turns);

    for (size_t t = 0; t < P->num_turns; t++)
    {
        stamina += stamina_list[t];
        if (stamina > P->stamina_max)
        {
            stamina = P->stamina_max;
        }

        if (demon_pos_idx < length_demon_idx_list)
        {
            Demon D = P->demons[demon_idx_list[demon_pos_idx]];

            if (stamina >= D.stamina_consumed)
            {
                stamina -= D.stamina_consumed;
                demon_pos_idx++;
                if (t + D.turn_recovered < P->num_turns)
                {
                    stamina_list[t + D.turn_recovered] += D.stamina_recovered;
                }

                for (size_t n = 0; n < D.num_turns_framgments; n++)
                {
                    if (t + n < P->num_turns)
                    {
                        fragments_list[t + n] += D.fragments[n];
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }

        fragments += fragments_list[t];
    }

    return fragments;
}

float randomFloat()
{
      float r = (float)rand()/(float)RAND_MAX;
      return r;
}
