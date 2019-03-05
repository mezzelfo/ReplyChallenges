#include "tool.h"

// Stampa la descrizione di un errore passata come stringa e termina il programma
void runtimeError(const char* errmsg)
{
    fprintf(stderr,"%s\n",errmsg);
    exit(EXIT_FAILURE);
}

// Leggi un intero da il file e controlla se la lettura è avvenuta correttamente
int readInt(FILE* f)
{
    int x;
    if(fscanf(f,"%d",&x)!=1)
        runtimeError("Errore durante la lettura del file");
    return x;
}

// Alloca un vettore di param:s byte e controlla se l'allocazione è avvenuta correttamente
void* alloca(size_t s)
{
    void* ptr = malloc(s);
    if(ptr==NULL)
        runtimeError("Errore durante l'allocazione della memoria");
    return ptr;
}

