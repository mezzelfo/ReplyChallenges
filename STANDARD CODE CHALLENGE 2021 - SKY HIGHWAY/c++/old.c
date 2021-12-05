/*void read_file(const char *filename, int ****buildings_map, int ***antennas, int *W, int *H, int *N, int *M, int *R)
{
    FILE *fileptr = fopen(filename, "r");
    if (!fileptr)
    {
        printf("ERROR: File %s not found\n", filename);
        exit(-1);
    }

    fscanf(fileptr, "%d %d %d %d %d", W, H, N, M, R);

    //Allocation
    *buildings_map = (int***) malloc(*W * sizeof(int **));
    for (int x = 0; x < *W; x++)
    {
        (*buildings_map)[x] = (int**) malloc(*H * sizeof(int *));
        for (int y = 0; y < *H; y++)
        {
            (*buildings_map)[x][y] = (int*) malloc(2 * sizeof(int));
        }
    }

    *antennas = (int**) malloc(*M * sizeof(int *));
    for (int a = 0; a < *M; a++)
    {
        (*antennas)[a] = (int*) malloc(2 * sizeof(int));
    }
    
    for (int i = 0; i < *N; i++)
    {
        int x, y, l, c;
        fscanf(fileptr, "%d %d %d %d\n", &x, &y, &l, &c);
        (*buildings_map)[x][y][0] = l;
        (*buildings_map)[x][y][1] = c;
    }
    for (int i = 0; i < *M; i++)
    {
        int r, c;
        fscanf(fileptr, "%d %d\n", &r, &c);
        (*antennas)[i][0] = r;
        (*antennas)[i][1] = c;
    }

    fclose(fileptr);
}*/