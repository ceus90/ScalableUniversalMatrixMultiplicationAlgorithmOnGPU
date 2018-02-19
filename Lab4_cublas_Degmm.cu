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
#include <cublas_v2.h>

extern "C" void launch_multiply(int bsize, int n, double *A, double *B, double *C)
{
double *Ad, *Bd, *Cd;

//Allocating device memory on the GPU for the matrices
cudaMalloc(&Ad,(size_t)(bsize*bsize*sizeof(double)));
cudaMalloc(&Bd,(size_t)(bsize*bsize*sizeof(double)));
cudaMalloc(&Cd,(size_t)(bsize*bsize*sizeof(double)));

//Copy A and B from host memory to device memory
cudaMemcpy(Ad,A,bsize*bsize*sizeof(double),cudaMemcpyHostToDevice);
cudaMemcpy(Bd,B,bsize*bsize*sizeof(double),cudaMemcpyHostToDevice);

int lda = n, ldb = n, ldc = n;
const double alf = 1;
const double bet = 0;
const double *alpha = &alf;
const double *beta = &bet;
cublasHandle_t handle;
cublasCreate(&handle);
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, Ad, lda, Bd, ldb, beta, Cd, ldc);
cublasDestroy(handle);

cudaFree(Ad);
cudaFree(Bd);
cudaFree(Cd);
}
