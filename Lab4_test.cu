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
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mat_mul(int bsize, int N, double *Ad, double *Bd, double *Cd)
{
int         m = blockIdx.x;
int         n = blockIdx.y;
int         i = threadIdx.x;
int         j = threadIdx.y;
int         k, p;
double c = 0.0;

__shared__  double As[32][32];
__shared__  double Bs[32][32];

for(p = 0; p < N/bsize; p++) {
As[i][j] = Ad[(m*bsize+i)*N+(p*bsize+j)];
Bs[i][j] = Bd[(p*bsize+i)*N+(n*bsize+j)];
__syncthreads();
for(k = 0; k < bsize; k++) {
c += As[i][k] * Bs[k][j];
}
}
Cd[(n*bsize+i)*N+(n*bsize+j)] = c;
}

extern "C" void launch_multiply(int bsize, int n, double *A, double *B, double *C)
{
double *Ad, *Bd, *Cd;
dim3   blockDim(bsize,bsize);
dim3   gridDim(n/bsize,n/bsize);

//Allocating device memory on the GPU for the matrices
cudaMalloc(&Ad,(size_t)(bsize*bsize*sizeof(double)));
cudaMalloc(&Bd,(size_t)(bsize*bsize*sizeof(double)));
cudaMalloc(&Cd,(size_t)(bsize*bsize*sizeof(double)));

//Copy A and B from host memory to device memory
cudaMemcpy(Ad,A,bsize*bsize*sizeof(double),cudaMemcpyHostToDevice);
cudaMemcpy(Bd,B,bsize*bsize*sizeof(double),cudaMemcpyHostToDevice);
mat_mul<<<gridDim,blockDim>>>(bsize,n,Ad,Bd,Cd);
for(i = 0; i < bsize*bsize-1; i++) {//Every element of C should be equal to bsize+1.
    if( fabs(C[i] - (bsize+1)) > 1e-6 ) printf("Incorrect result:%f\n", C[i]);
cudaFree(Ad);
cudaFree(Bd);
cudaFree(Cd);
}
