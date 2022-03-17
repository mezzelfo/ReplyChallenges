#ifndef TOOLS_H
#define TOOLS_H

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

Problem parse_input_file(const char *filename);
long unsigned int complete_simulator(const int length_demon_idx_list, const int *demon_idx_list, const Problem *P);

#endif