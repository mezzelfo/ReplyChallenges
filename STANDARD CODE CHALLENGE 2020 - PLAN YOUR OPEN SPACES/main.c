#include <stdio.h>
#include <stdlib.h>
#include "sortedLinkedList.h"

#define MAXLENGTH 50

typedef struct
{
    int pos;
    unsigned company;
    unsigned bonus;
    unsigned* skills;
    short skills_num;
    short isManager;
} Replayer;



unsigned valuesDictChar(const char c)
{
    unsigned val;
    if((c >= 48)&&(c<=57))
        val = c-48;
    if((c >= 97)&&(c<=122))
        val = c-97+10;
    if (val >= 36)
        fprintf(stderr,"Errore hash carattere %c\n",c);
    return val;    
}

unsigned hashString(const char* const buff)
{
    return   valuesDictChar(buff[0])
            +valuesDictChar(buff[1])*36
            +valuesDictChar(buff[2])*36*36
            +valuesDictChar(buff[3])*36*36*36
            +valuesDictChar(buff[4])*36*36*36*36
            +valuesDictChar(buff[5])*36*36*36*36*36;
}

int compareUnsigned( const void* arg1, const void* arg2 )
{
    if (*(unsigned*)arg1 < *(unsigned*)arg2) return -1;
    if (*(unsigned*)arg1 > *(unsigned*)arg2) return 1;
    return 0;
}

unsigned common_element_sorted(unsigned* v1, int s1, unsigned* v2, int s2)
{
    unsigned num_common_elements = 0;
    unsigned index1 = 0, index2 = 0;
    while (1)
    {
        if (v1[index1] == v2[index2])
        {
            num_common_elements++;
            index1++;
            index2++;
        }
        else if(v1[index1] >= v2[index2])
        {
            index1++;
        }
        else
        {
            index2++;
        }
        if ((index1 >= s1) || (index2 >= s2)) return num_common_elements;
    }
}

unsigned totpow(Replayer* r1, Replayer* r2)
{
    unsigned company_bonus = 0;
    if (r1->company == r2->company) company_bonus = r1->bonus*r2->bonus;
    if (r1->isManager || r2->isManager) return company_bonus;
    unsigned common_skills = 0;
    common_skills = common_element_sorted(r1->skills,r1->skills_num,r2->skills,r2->skills_num);
    unsigned distinct_skills = r1->skills_num+r2->skills_num-2*common_skills;
    return common_skills*distinct_skills+company_bonus;
}

int usablePair(Pair* p)
{
    if(p==NULL) return -1;
    Replayer* r1 = (Replayer*)(p->r1);
    Replayer* r2 = (Replayer*)(p->r2);
    if (r1->pos==-1 && r2->pos==-1) return 1;
    return 0;
}

int main(int argc, char const *argv[])
{
    if (argc != 3) fprintf(stderr,"Utilizzo: ./a.out <nomefileinput> <nomefileoutput>\n");
    FILE* inputFile = fopen(argv[1],"r");
    if (inputFile == NULL) fprintf(stderr,"Impossibile aprire file %s\n",argv[1]);

    unsigned W,H;
    if(fscanf(inputFile,"%d %d\n",&W,&H)!=2) fprintf(stderr,"Errore parsing 1\n");
    printf("W:%d\tH:%d\n",W,H);

    char* matrix = (char*) calloc((W+2)*(H+2),sizeof(char));
    if(!matrix) fprintf(stderr,"Errore allocazione matricepavimento\n");
    
    for(int i=1; i<=H; i++)
    {
        if(fscanf(inputFile,"%s\n",matrix) != 1) fprintf(stderr,"Errore parsing 2\n");
        sprintf(matrix+(i*(W+2)),"#%s#",matrix);
    }
    for(int i=0; i<W+2; i++)
    {
        matrix[i] = '#';
        matrix[(H+1)*(W+2)+i] = '#';
    }
    
    unsigned D;
    if(fscanf(inputFile,"%d\n",&D)!=1) fprintf(stderr,"Errore parsing 3\n");
    Replayer* developers = (Replayer*)calloc(D,sizeof(Replayer));
    for(int i=0; i<D; i++)
    {
        developers[i].isManager = 0;
        developers[i].pos = -1;
        char companyBuf[8];
        if(fscanf(inputFile,"%s %d %hd ",
                    companyBuf,
                    &(developers[i].bonus),
                    &(developers[i].skills_num)) != 3) fprintf(stderr,"errore 4 parsing\n");
        developers[i].company = hashString(companyBuf);

        char skillBuf[8];
        developers[i].skills = (unsigned*)calloc(developers[i].skills_num,sizeof(unsigned));
        for(int j=0; j<developers[i].skills_num; j++)
        {
            if(fscanf(inputFile,"%s",skillBuf)!=1) fprintf(stderr,"errore 5 parsing\n");
            developers[i].skills[j] = hashString(skillBuf);
        }
        qsort(developers[i].skills,developers[i].skills_num,sizeof(unsigned),compareUnsigned);
    }

    unsigned M;
    if(fscanf(inputFile,"%d\n",&M)!=1) fprintf(stderr,"Errore parsing 6\n");
    Replayer* managers = (Replayer*)calloc(M,sizeof(Replayer));
    for(int i=0; i<M; i++)
    {
        managers[i].isManager = 1;
        managers[i].pos = -1;
        char companyBuf[8];
        if(fscanf(inputFile,"%s %d\n",
                    companyBuf,
                    &(managers[i].bonus))!= 2) fprintf(stderr,"errore 7 parsing\n");
        managers[i].company = hashString(companyBuf);
        managers[i].skills = NULL;
        managers[i].skills_num = 0;
    }
    if (getc(inputFile) != EOF)
        fprintf(stderr, "Errore generico parsing file input\n");
    else
        fclose(inputFile);
    
    printf("D:%d\tM:%d\n",D,M);

    // DEBUG
    List* DDqueueDebug = getNewList(MAXLENGTH);
    unsigned* best = (unsigned*)calloc(D,sizeof(unsigned));
    for(int i=0; i<D; i++)
    {
        for(int j=i+1; j<D; j++)
        {
            Replayer* r1 = &(developers[i]);
            Replayer* r2 = &(developers[j]);
            unsigned p = totpow(r1,r2);
            if (best[i] < p) best[i] = p;
        }
        if (i%500 == 0) printf("%d\n",i);//cout << i << endl;
    }
    unsigned long long pwr = 0;
    for(int i=0; i<D; i++) pwr += best[i];
    free(best);
    printf("Total Skill PWR: %llu\n",4*pwr);
    return -1;
    //END DEBUG

    List* DDqueue = getNewList(MAXLENGTH);
    for(int i=0; i<D; i++)
    {
        for(int j=i+1; j<D; j++)
        {
            Replayer* r1 = &(developers[i]);
            Replayer* r2 = &(developers[j]);
            addToList(DDqueue,r1,r2,totpow(r1,r2));
        }
    }
    

    List* MMqueue = getNewList(MAXLENGTH);
    for(int i=0; i<M; i++)
    {
        for(int j=i+1; j<M; j++)
        {
            Replayer* r1 = &(managers[i]);
            Replayer* r2 = &(managers[j]);
            addToList(MMqueue,r1,r2,totpow(r1,r2));
        }
    }

    List* DMqueue = getNewList(MAXLENGTH);
    for(int i=0; i<D; i++)
    {
        for(int j=0; j<M; j++)
        {
            Replayer* r1 = &(developers[i]);
            Replayer* r2 = &(managers[j]);
            addToList(DMqueue,r1,r2,totpow(r1,r2));
        }
    }


    printf("Ho creato le tre code\n");

    DDqueue->iterator = DDqueue->head;
    MMqueue->iterator = MMqueue->head;
    DMqueue->iterator = DMqueue->head;

    const short deltas[] = {+1,-1,-(W+2),+(W+2)};
    for(int row=1; row<=H; row++)
    {
        for(int col=1; col<=H; col++)
        {
            unsigned pos1 = row*(W+2)+col;
            char here = matrix[pos1];
            if (here == '#') continue;
            for(int k=0; k<4; k++)
            {   
                unsigned pos2 = row*(W+2)+col+deltas[k];
                char there = matrix[pos2];
                List* queue = NULL;
                if (there=='#') continue;
                else if (here=='_' && there=='_') queue = DDqueue;
                else if (here=='M' && there=='M') queue = MMqueue;
                else if ((here=='_' && there=='M')||(here=='M' && there=='_')) queue = DMqueue;
                else fprintf(stderr,"Errore mappa: here=<%c>, there=<%c>\n",here,there);
                while (!usablePair(queue->iterator)) queue->iterator = queue->iterator->next;
                if (queue->iterator == NULL) continue;
                if (queue != DMqueue)
                {
                    //Vedere quale delle due pos è più conveniente
                    Replayer* r1 = (Replayer*) queue->iterator->r1;
                    Replayer* r2 = (Replayer*) queue->iterator->r2;
                    r1->pos = pos1;
                    r2->pos = pos2;
                } else {
                    Replayer* r1 = (Replayer*) queue->iterator->r1;
                    Replayer* r2 = (Replayer*) queue->iterator->r2;
                    if ((r1->isManager && here=='M') || (r2->isManager && there=='M'))
                    {
                        r1->pos = pos1;
                        r2->pos = pos2;
                    } else 
                    {
                        r1->pos = pos2;
                        r2->pos = pos1;
                    }
                }

                matrix[pos1] = '#';
                matrix[pos2] = '#';
                break; //continue?
            }
        }
    }


    printf("Inizio creazione file di output\n");
    FILE* outputFile = fopen(argv[2],"w");
    for(int i=0; i<D; i++)
    {
        Replayer* r = &(developers[i]); 
        if(r->pos != -1)
        {
            int x = r->pos%(W+2)-1;
            int y = (r->pos-x)/(W+2)-1;
            if((y+1)*(W+2)+(x+1) != r->pos) fprintf(stderr,"Errore indici (%d,%d)->%d\n",x,y,r->pos);
            fprintf(outputFile,"%d %d\n",x,y);
        }
        else fprintf(outputFile,"X\n");            
    }

    for(int i=0; i<M; i++)
    {
        Replayer* r = &(managers[i]); 
        if(r->pos != -1)
        {
            int x = r->pos%(W+2)-1;
            int y = (r->pos-x)/(W+2)-1;
            if((y+1)*(W+2)+(x+1) != r->pos) fprintf(stderr,"Errore indici (%d,%d)->%d\n",x,y,r->pos);
            fprintf(outputFile,"%d %d\n",x,y);
        }
        else fprintf(outputFile,"X\n");            
    }
    fclose(outputFile);
    
    

    free(matrix);
    for(int i=0; i<D; i++) free(developers[i].skills);
    free(developers);

    clearList(DDqueue);
    clearList(MMqueue);
    clearList(DMqueue);

    return 0;
}
