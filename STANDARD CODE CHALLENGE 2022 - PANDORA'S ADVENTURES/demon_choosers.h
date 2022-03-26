#ifndef DEMON_CHOOSERS_H
#define DEMON_CHOOSERS_H

#include "utils.h"

typedef struct
{
    size_t length;
    size_t demon_pos_idx;
    int *list_ids;
} DemonList;

Demon demon_chooser_from_list(unsigned int stamina, size_t turn, const unsigned int *stamina_list, const Demon *defeated_demons_list, const Problem *P, void *args);
#endif