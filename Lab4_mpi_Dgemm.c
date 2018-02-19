#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include <math.h>
#include <stddef.h>
#include <sys/time.h>
#include <mpi.h>

void launch_multiply(int bsize, int n, double *A, double *B, double *C);

int min(int a, int b)
{
    return a < b ? a : b;
}

double get_cur_time();

int main(int argc, char** argv)
{
    int i, j, k, x, chunk, n = 1024, bsize = 32, size, rank, proc, start = 0, finish = 0;
    double btime, etime, *Ad, *Bd, *Cd;
    double *A = malloc(n*n*sizeof(double));
    double *B = malloc(n*n*sizeof(double));
    double *C = malloc(n*n*sizeof(double));
    
    srand((unsigned)time(NULL));
    for(i = 0; i < n*n; i++)        //Fill matrices A, B and C with random numbers
    {
        A[i] = (double)rand()/(double)RAND_MAX;
        B[i] = (double)rand()/(double)RAND_MAX;
        C[i] = 0.0;
    }
    MPI_Init(&argc, &argv); //starts MPI
    MPI_Comm_size(MPI_COMM_WORLD, &size);   //get number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   //get current process id
    MPI_Status status;
    printf("\nHello, I am process %d among %d processes\n", rank, size);
    chunk = n/bsize;
    if (rank == 0)
    {       //Sending Matrix A and B to different processes, according to the block size
        printf("\nSending now\n");
        for (proc = 1; proc < size; proc++)
        {
            double *tmp = malloc(bsize*bsize*sizeof(double));
            for (j = 0; j < bsize; j++)
                for (i = 0; i < bsize; i++)
                    tmp[i*bsize+j] = A[i*bsize+j];
            MPI_Send(tmp, bsize*bsize, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
            free(tmp);
        }
        
        for (proc = 1; proc < size; proc++)
        {
            double *tmp = malloc(bsize*bsize*sizeof(double));
            for (i = 0; i < bsize; i++)
                for (j = 0; j < bsize; j++)
                    tmp[i*bsize+j] = B[i*bsize+j];
            MPI_Send(tmp, bsize*bsize, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD);
            free(tmp);
        }
    }
    else
    {   //Receiving Matrix A and B from different processes, in their block size
        MPI_Recv(A, bsize*bsize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(B, bsize*bsize, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
        btime = get_cur_time();
        launch_multiply(bsize,n,A,B,C);
        etime = get_cur_time();
    }
    printf("\nElapsed time = %f seconds\n",etime-btime);
    MPI_Finalize();
    free(A);
    free(B);
    free(C);
    return 0;
}

double get_cur_time()
{
    struct timeval   tv;
    struct timezone  tz;
    double cur_time;
    
    gettimeofday(&tv, &tz);
    cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;
    
    return cur_time;
}
