#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct
{
    int id;
    unsigned int stamina_consumed, turn_recovered, stamina_recovered;
    size_t num_turns_framgments;
    unsigned int *fragments;
} Demon;

typedef struct
{
    unsigned int stamina_init, stamina_max, num_turns;
    size_t num_demons;
    Demon *demons;
} Problem;

Problem parse_input_file(const char *filename);
long unsigned int complete_simulator(
    const Demon (*demon_chooser)(
        unsigned int stamina, size_t turn, const unsigned int *stamina_list, const Demon *defeated_demons_list, const Problem *P, void *args),
    const Problem *P, void *demont_chooser_args);
float randomFloat();

#endif