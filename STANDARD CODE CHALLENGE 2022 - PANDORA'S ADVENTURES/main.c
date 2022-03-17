#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    unsigned int id, stamina_consumed, turn_recovered, stamina_recovered;
    size_t num_turns_framgments;
    unsigned int *fragments;
} Demon;

typedef struct
{
    unsigned int stamina_init, stamina_max, num_turns;
    size_t num_demons;
    Demon *demons;
} Problem;

Problem parse_input_file(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    Problem P;
    fscanf(fp, "%d %d %d %ld\n",
           &(P.stamina_init),
           &(P.stamina_max),
           &(P.num_turns),
           &(P.num_demons));

    P.demons = (Demon *)malloc(sizeof(Demon) * P.num_demons);
    for (size_t i = 0; i < P.num_demons; i++)
    {
        fscanf(fp, "%d %d %d %ld ",
               &(P.demons[i].stamina_consumed),
               &(P.demons[i].turn_recovered),
               &(P.demons[i].stamina_recovered),
               &(P.demons[i].num_turns_framgments));
        P.demons[i].fragments = (unsigned int *)malloc(sizeof(unsigned int) * P.demons[i].num_turns_framgments);
        for (size_t j = 0; j < P.demons[i].num_turns_framgments; j++)
        {
            fscanf(fp, "%d", &(P.demons[i].fragments[j]));
        }
        fscanf(fp, "\n");
    }

    return P;
}

long unsigned int complete_simulator(const int length_demon_idx_list, const int *demon_idx_list, const Problem *P)
{
    unsigned int stamina = P->stamina_init;
    unsigned int fragments = 0;
    size_t demon_pos_idx = 0;

    unsigned int* fragments_list = (unsigned int *)malloc(sizeof(unsigned int) * P->num_turns);
    unsigned int* stamina_list = (unsigned int *)malloc(sizeof(unsigned int) * P->num_turns);

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

int main(int argc, char const *argv[])
{
    Problem prob = parse_input_file(argv[1]);

    unsigned int* list = (unsigned int*) malloc(sizeof(unsigned int)*prob.num_demons);
    
    long unsigned int reward = complete_simulator(prob.num_demons,list,&prob);

    printf("Input file %s. Reward %ld\n", argv[1], reward);

    return 0;
}
