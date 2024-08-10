#include<stdio.h>
#include<iostream>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
__global__ void MatMulKernel(Matrix, Matrix, Matrix);
void MatMul(Matrix A, Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);

    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
    cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,cudaMemcpyHostToDevice);
    
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + BLOCK_SIZE - 1) / dimBlock.x, 
                 (A.height + BLOCK_SIZE - 1) / dimBlock.y); 
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Wait for the GPU to finish before accessing the results on the host
    // cudaDeviceSynchronize(); -> not required because cudaMemcpy after this blocks further execution
    // until previous kernel (MatMulKernel) is completly executed

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
    cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
    C.elements[1] = 20;
}

void dispMatrix(Matrix X,std::string name){
    std::cout<<"Mat "<<name<<std::endl;
    for(int i = 0; i < X.height; i++) {
        for (int j = 0; j < X.height; j++) {
            std::cout<<X.elements[i*X.height + j]<<" ";            
        }
        std::cout<<std::endl;
    }
}

int main(){
    int N = 5;
    
     // Declare and initialize host matrices
    Matrix A, B;
    A.width = N;
    A.height = N;

    B.width = N;
    B.height = N;

    Matrix C;
    C.width = N;
    C.height = N;

    // Allocate memory for matrices A and B
    A.elements = (float*)malloc(N * N * sizeof(float));
    B.elements = (float*)malloc(N * N * sizeof(float));
    C.elements = (float*)malloc(N * N * sizeof(float));


    // Initialize matrices A and B
    for(int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A.elements[i*A.height + j] = i + j;
            B.elements[i*B.height + j] = i * j;
        }
    }

    MatMul(A, B, C);

    dispMatrix(A,"A");
    dispMatrix(B,"B");
    dispMatrix(C,"C");

}
