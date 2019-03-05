#include <stdio.h>
#include <stdlib.h>

// Stampa la descrizione di un errore passata come stringa e termina il programma
void runtimeError(const char* errmsg);

// Leggi un intero da il file e controlla se la lettura è avvenuta correttamente
int readInt(FILE* f);

// Alloca un vettore di param:s byte e controlla se l'allocazione è avvenuta correttamente
void* alloca(size_t s);

