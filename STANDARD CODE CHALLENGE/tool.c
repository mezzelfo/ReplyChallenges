#include "tool.h"

void runtimeError(const char* errmsg)
{
	fprintf(stderr, "%s\n", errmsg);
	exit(EXIT_FAILURE);
}

void* secMalloc(const size_t s)
{
	void* ptr = malloc(s);
	if (ptr == NULL)
		runtimeError("Errore durante l'allocazione della memoria");
	return ptr;
}

int readInt(FILE* f)
{
	int n;
	if(fscanf(f,"%d",&n) != 1)
		runtimeError("Errore durante la lettura del file");
	return n;
}

unsigned readUnsigned(FILE* f)
{
	unsigned n;
	if(fscanf(f,"%u",&n) != 1)
		runtimeError("Errore durante la lettura del file");
	return n;
}
