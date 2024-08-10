#include <stdio.h>
#include <iostream>
#define N 10000

// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],float C[N][N]){
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main(){
    float (*A)[N], (*B)[N], (*C)[N];

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&A, N*N*sizeof(float));
    cudaMallocManaged(&B, N*N*sizeof(float));
    cudaMallocManaged(&C, N*N*sizeof(float));

    //intialize 2d matrix
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            A[i][j] = i*2 + 4;
            B[i][j] = j*2 + i;
        }
    }

    // Kernel invocation with
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    
    // Wait for the GPU to finish before accessing the results on the host
    cudaDeviceSynchronize();

    std::cout<<"Result of matrix addition"<<std::endl;

    // for(int i=0; i<N; i++){
    //     for(int j=0; j<N; j++){
    //         std::cout<<"Idx "<<i<<","<<j<<": "<<C[i][j]<<std::endl;
    //     }
    // }

     // Free managed memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
