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
        P.demons[i].id = i;
        P.demons[i].fragments = (unsigned int *)malloc(sizeof(unsigned int) * P.demons[i].num_turns_framgments);
        for (size_t j = 0; j < P.demons[i].num_turns_framgments; j++)
        {
            checkfscaf = fscanf(fp, "%d", &(P.demons[i].fragments[j]));
            assert(checkfscaf == 1);
        }
        checkfscaf = fscanf(fp, "\n");
    }

    fclose(fp);
    return P;
}

long unsigned int complete_simulator(
    const Demon (*demon_chooser)(
        unsigned int stamina, size_t turn, const unsigned int *stamina_list, const Demon *defeated_demons_list, const Problem *P, void *args),
    const Problem *P, void *demont_chooser_args)
{
    unsigned int stamina = P->stamina_init;
    unsigned int fragments = 0;

    unsigned int *fragments_list = (unsigned int *)calloc(P->num_turns, sizeof(unsigned int));
    unsigned int *stamina_list = (unsigned int *)calloc(P->num_turns, sizeof(unsigned int));
    Demon *defeated_demons_list = (Demon *)malloc(sizeof(Demon) * P->num_turns);

    for (size_t t = 0; t < P->num_turns; t++)
    {
        stamina += stamina_list[t];
        if (stamina > P->stamina_max)
        {
            stamina = P->stamina_max;
        }

        defeated_demons_list[t].id = -1;
        Demon D = demon_chooser(stamina, t, stamina_list, defeated_demons_list, P, demont_chooser_args);
        if (D.id != -1)
        {
            printf("affronto %d. Per ora ho %d frammenti\n", D.id, fragments);
            assert(stamina >= D.stamina_consumed);
            defeated_demons_list[t] = D;
            stamina -= D.stamina_consumed;
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

        fragments += fragments_list[t];
    }

    return fragments;
}

float randomFloat()
{
    float r = (float)rand() / (float)RAND_MAX;
    return r;
}
