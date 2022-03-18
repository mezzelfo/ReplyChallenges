#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define NUM_ACTIONS (2)
typedef enum
{
    GOOD,
    BAD
} Action;

typedef struct
{
    unsigned int stamina, turn;
} State;

Action classify_demon(const Demon d)
{
    if (d.stamina_recovered > d.stamina_consumed)
    {
        return GOOD;
    }
    return BAD;
}

Demon get_demon_from_action(const Action A, const State S, const Problem *P)
{
    Demon d;
    do
    {
        d = P->demons[rand() % P->num_demons];
    } while (classify_demon(d) != A);
    return d;
}

void Q_value_simulator(const Action A, const State S, const Problem *P, State *new_state, unsigned int *reward)
{
    Demon D = get_demon_from_action(A, S, P);
    new_state->turn = S.turn + 1;
    new_state->stamina = S.stamina;
    if (S.stamina >= D.stamina_consumed)
    {
        new_state->stamina += D.stamina_recovered - D.stamina_consumed;
    }
    if (new_state->stamina > P->stamina_max)
    {
        new_state->stamina = P->stamina_max;
    }

    *reward = 0;

    for (size_t t = 0; t < D.num_turns_framgments; t++)
    {
        if (new_state->turn >= P->num_turns)
        {
            break;
        }
        *reward += D.fragments[t];
    }
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));

    Problem prob = parse_input_file(argv[1]);

    // TODO create pools of demons for file 03

    int minimum_stamina_consumed = +999999.9;
    for (size_t d = 0; d < prob.num_demons; d++)
    {
        if (minimum_stamina_consumed > prob.demons[d].stamina_consumed)
        {
            minimum_stamina_consumed = prob.demons[d].stamina_consumed;
        }
    }

    // Q is the Q-values table
    // TODO "linearize" the array for speed
    float ***Q = (float ***)malloc(sizeof(float **) * NUM_ACTIONS);
    for (size_t a = 0; a < NUM_ACTIONS; a++)
    {
        Q[a] = (float **)malloc(sizeof(float *) * prob.num_turns);
        for (size_t t = 0; t < prob.num_turns; t++)
        {
            Q[a][t] = (float *)calloc(prob.stamina_max + 1, sizeof(float));
        }
    }

    const float alpha = 0.05;
    const float gamma = 0.99;
    const float epsilon = 0.05;
    int counter = 0;

    for (size_t iterations = 0; iterations < 5*1000000; iterations++)
    {
        counter++;
        if (counter % 10000 == 0)
        {
            printf("counter %d\n", counter);
        }
        State s;
        if (randomFloat() < 0.5)
        {
            s.stamina = prob.stamina_init;
            s.turn = 0;
        }
        else
        {
            s.stamina = rand() % (prob.stamina_max+1);
            s.turn = rand() % (prob.num_turns-10);
        }

        for (size_t t = 0; t < prob.num_turns; t++)
        {
            // TODO choose an action: exploration vs exploitation
            Action a;
            if (randomFloat() < epsilon)
            {
                if (randomFloat() < 0.5)
                    a = GOOD;
                else
                    a = BAD;
            }
            else
            {
                if (Q[GOOD][s.turn][s.stamina] > Q[BAD][s.turn][s.stamina])
                    a = GOOD;
                else
                    a = BAD;
            }

            State new_state;
            unsigned int reward;
            Q_value_simulator(a, s, &prob, &new_state, &reward);

            if (new_state.turn == prob.num_turns)
                break;

            float Qmax = -9999.9;
            for (size_t next_action = 0; next_action < NUM_ACTIONS; next_action++)
            {
                if (Qmax < Q[next_action][new_state.turn][new_state.stamina])
                {
                    Qmax = Q[next_action][new_state.turn][new_state.stamina];
                }
            }

            Q[a][s.turn][s.stamina] += alpha * (reward + gamma * Qmax - Q[a][s.turn][s.stamina]);

            s = new_state;

            if (s.stamina < minimum_stamina_consumed)
                break;
        }
    }

    // Save Q table
    FILE *fp = fopen("Q_table", "w");
    fprintf(fp, "{");
    for (size_t a = 0; a < NUM_ACTIONS; a++)
    {
        if (a == 0)
            fprintf(fp, "{");
        else
            fprintf(fp, ",{");
        for (size_t t = 0; t < prob.num_turns; t++)
        {
            if (t == 0)
                fprintf(fp, "{");
            else
                fprintf(fp, ",{");
            for (size_t s = 0; s < prob.stamina_max; s++)
            {
                if (s == 0)
                {
                    fprintf(fp, "%f", Q[a][t][s]);
                }
                else
                {
                    fprintf(fp, ",%f", Q[a][t][s]);
                }
            }
            fprintf(fp, "}");
        }
        fprintf(fp, "}");
    }
    fprintf(fp, "}");

    // int list[] = {1,3,2,0,4};

    // long unsigned int reward = complete_simulator(prob.num_demons, list, &prob);

    // printf("Input file %s. Reward %ld\n", argv[1], reward);

    return 0;
}
