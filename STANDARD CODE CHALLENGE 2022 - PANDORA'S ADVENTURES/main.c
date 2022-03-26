#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include "demon_choosers.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

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

Demon *good_demons_pool, *bad_demons_pool;
size_t good_demons_pool_length, bad_demons_pool_length;

Demon get_demon_from_action(const Action A, const State S, const Problem *P)
{
    Demon *chosen_demon_pool;
    size_t chosen_demon_pool_length;
    if (A == GOOD)
    {
        chosen_demon_pool = good_demons_pool;
        chosen_demon_pool_length = good_demons_pool_length;
    }
    else
    {
        chosen_demon_pool = bad_demons_pool;
        chosen_demon_pool_length = bad_demons_pool_length;
    }
    Demon d;
    for (size_t i = 0; i < 100; i++)
    {
        d = chosen_demon_pool[rand() % chosen_demon_pool_length];
        if (d.stamina_consumed < S.stamina)
            break;
    }
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

Demon demon_chooser_from_Qvalues(unsigned int stamina, size_t turn, const unsigned int *stamina_list, const Demon *defeated_demons_list, const Problem *P, void *args)
{
    const float *const Q = (float *)args;
    float Q_GOOD = Q[turn * (P->stamina_max + 1) * NUM_ACTIONS + stamina * NUM_ACTIONS + GOOD];
    float Q_BAD = Q[turn * (P->stamina_max + 1) * NUM_ACTIONS + stamina * NUM_ACTIONS + BAD];
    State s;
    s.stamina = stamina;
    s.turn = turn;
    Action a;
    if (Q_GOOD > Q_BAD)
        a = GOOD;
    else
        a = BAD;

    Demon d = get_demon_from_action(a, s, P);
    if (d.stamina_consumed < stamina)
        return d;

    Demon nullDemon;
    nullDemon.id = -1;
    return nullDemon;
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));

    Problem prob = parse_input_file(argv[1]);

    // DemonList demonList;
    // demonList.demon_pos_idx = 0;
    // demonList.length = prob.num_demons;
    // demonList.list_ids = (int*)malloc(sizeof(int)*demonList.length);
    // FILE *fpp = fopen("submit_3.txt", "r");
    // for (size_t d = 0; d < prob.num_demons; d++)
    // {
    //     fscanf(fpp,"%d\n",&(demonList.list_ids[d]));
    // }
    // fclose(fpp);
    // long unsigned int reward = complete_simulator(&demon_chooser_from_list, &prob, (void*)(&demonList));
    // printf("%ld\n",reward);
    // return 0;

    int minimum_stamina_consumed = +999999.9;
    good_demons_pool = (Demon *)malloc(sizeof(Demon) * prob.num_demons);
    bad_demons_pool = (Demon *)malloc(sizeof(Demon) * prob.num_demons);
    good_demons_pool_length = 0;
    bad_demons_pool_length = 0;
    for (size_t d_idx = 0; d_idx < prob.num_demons; d_idx++)
    {
        Demon d = prob.demons[d_idx];
        if (d.turn_recovered > 10)
            continue;

        if (d.stamina_recovered > d.stamina_consumed)
        {
            good_demons_pool[good_demons_pool_length] = d;
            good_demons_pool_length++;
        }
        else
        {
            bad_demons_pool[bad_demons_pool_length] = d;
            bad_demons_pool_length++;
        }

        if (minimum_stamina_consumed > d.stamina_consumed)
        {
            minimum_stamina_consumed = d.stamina_consumed;
        }
    }

    good_demons_pool = realloc(good_demons_pool, sizeof(Demon) * good_demons_pool_length);
    bad_demons_pool = realloc(bad_demons_pool, sizeof(Demon) * bad_demons_pool_length);

    assert(good_demons_pool != NULL);
    assert(bad_demons_pool != NULL);

    // Q is the Q-values table
    // Q[turn][stamina][action]
    // matrix[ i ][ j ][ k ] : L by N by M = array[ i*(N*M) + j*M + k ]
    float *Q = (float *)calloc(prob.num_turns * (prob.stamina_max + 1) * NUM_ACTIONS, sizeof(float));

    assert(Q != NULL);

    const float alpha = 0.05;
    const float epsilon = 0.75;
    int counter = 0;

    for (size_t iterations = 0; iterations < 1000000; iterations++)
    {
        counter++;
        if (counter % 10000 == 0)
        {
            printf("counter %d\n", counter);
        }
        State s;
        if (randomFloat() < 0.1)
        {
            s.stamina = prob.stamina_init;
            s.turn = 0;
        }
        else
        {
            s.stamina = rand() % (prob.stamina_max + 1);
            s.turn = rand() % (prob.num_turns - 10);
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
                // Q[turn][stamina][action]
                // matrix[ i ][ j ][ k ] : L by N by M = array[ i*(N*M) + j*M + k ]
                float Q_GOOD = Q[s.turn * (prob.stamina_max + 1) * NUM_ACTIONS + s.stamina * NUM_ACTIONS + GOOD];
                float Q_BAD = Q[s.turn * (prob.stamina_max + 1) * NUM_ACTIONS + s.stamina * NUM_ACTIONS + BAD];
                if (Q_GOOD > Q_BAD)
                    a = GOOD;
                else
                    a = BAD;
            }

            State new_state;
            unsigned int reward;
            Q_value_simulator(a, s, &prob, &new_state, &reward);

            if (new_state.turn == prob.num_turns)
                break;

            float Q_GOOD = Q[new_state.turn * (prob.stamina_max + 1) * NUM_ACTIONS + new_state.stamina * NUM_ACTIONS + GOOD];
            float Q_BAD = Q[new_state.turn * (prob.stamina_max + 1) * NUM_ACTIONS + new_state.stamina * NUM_ACTIONS + BAD];
            float Qmax = MAX(Q_GOOD, Q_BAD);

            Q[s.turn * (prob.stamina_max + 1) * NUM_ACTIONS + s.stamina * NUM_ACTIONS + a] +=
                alpha * (reward + Qmax - Q[s.turn * (prob.stamina_max + 1) * NUM_ACTIONS + s.stamina * NUM_ACTIONS + a]);

            s = new_state;

            if (s.stamina < minimum_stamina_consumed)
                break;
        }
    }

    long unsigned int reward = complete_simulator(&demon_chooser_from_Qvalues, &prob, (void *)(Q));
    printf("Q_values reward %ld\n", reward);

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
                    fprintf(fp, "%f", Q[t * (prob.stamina_max + 1) * NUM_ACTIONS + s * NUM_ACTIONS + a]);
                }
                else
                {
                    fprintf(fp, ",%f", Q[t * (prob.stamina_max + 1) * NUM_ACTIONS + s * NUM_ACTIONS + a]);
                }
            }
            fprintf(fp, "}");
        }
        fprintf(fp, "}");
    }
    fprintf(fp, "}");

    free(good_demons_pool);
    free(bad_demons_pool);
    free(Q);
    return 0;
}
