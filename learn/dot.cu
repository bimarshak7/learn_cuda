#include <iostream>
#include <math.h>


__global__
void dot(int n, float *x, float *y, float *z){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i+=stride){
    z[i] = x[i] * y[i];
  }
}

int main(){
  int N = 1000;
  float *x, *y, *z;

 // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&z, N*sizeof(float));
  
  for(int i=0; i<N; i++){
    x[i] = i;
    y[i] = N-i-1;
  }

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  dot<<<numBlocks, blockSize>>>(N, x, y,z);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  for(int i = 0; i < N; i++){
    std::cout<<z[i]<<',';
  }  
  std::cout<<std::endl;

}
