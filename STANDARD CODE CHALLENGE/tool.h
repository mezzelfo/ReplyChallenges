#ifndef TOOL_H
#define TOOL_H
#include <stdio.h>
#include <stdlib.h>

// Stampa un errore e termina il programma
void throwError(const char* errmsg);

// Allocazione della memora e controllo se è avvenuta con successo
void* secMalloc(const size_t s);

// Legge un intero da un file e controlla se ci sono stati errori
int readInt(FILE* f);

// Legge un unsigned da un file e controlla se ci sono stati errori
unsigned readUnsigned(FILE* f);

// Legge un unsigned da un file e controlla se ci sono stati errori
char readChar(FILE* f);

#endif