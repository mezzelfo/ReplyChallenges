#include <stdio.h>
#include <stdlib.h>

void throwError(const char* errmsg)
{
	fprintf(stderr, "%s\n", errmsg);
	exit(EXIT_FAILURE);
}

void* secMalloc(const size_t s)
{
	void* ptr = malloc(s);
	if (ptr == NULL)
		throwError("Errore durante l'allocazione della memoria");
	return ptr;
}

int readInt(FILE* f)
{
	int n;
	if(fscanf(f,"%d",&n) != 1)
		throwError("Errore durante la lettura del file");
	return n;
}

unsigned readUnsigned(FILE* f)
{
	unsigned n;
	if(fscanf(f,"%u",&n) != 1)
		throwError("Errore durante la lettura del file");
	return n;
}

char readChar(FILE* f)
{
	char c;
	do
	{
		if(fscanf(f,"%c",&c) != 1)
			throwError("Errore durante la lettura del file");
	}while((c == 13) || (c == 10));
	return c;
}
