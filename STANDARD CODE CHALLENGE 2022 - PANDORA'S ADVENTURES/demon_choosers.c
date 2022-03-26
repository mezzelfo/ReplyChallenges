#include "demon_choosers.h"

Demon demon_chooser_from_list(unsigned int stamina, size_t turn, const unsigned int *stamina_list, const Demon *defeated_demons_list, const Problem *P, void *args)
{
    Demon nullDemon;
    nullDemon.id = -1;

    DemonList *demonList = (DemonList *)args;
    size_t next_demon_pos = demonList->demon_pos_idx;
    if (next_demon_pos >= demonList->length)
        return nullDemon;
    Demon D = P->demons[(demonList->list_ids)[next_demon_pos]];
    if (D.stamina_consumed > stamina)
        return nullDemon;
    demonList->demon_pos_idx = 1 + demonList->demon_pos_idx;
    return D;
}