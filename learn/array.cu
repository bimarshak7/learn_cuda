#include <iostream>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y, float *result)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    result[i] = x[i] + y[i];
}

// Kernel function to perform dot product of two arrays
__global__
void dot_arr(int n, float *x, float *y, float *result)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    result[i] = x[i] * y[i];
}


class Array {
private:
    float* data;
    int size;

public:
    // Constructor to initialize the array with a given 1D array and size
    Array(float arr[], int n) {
        size = n;
        // data = new float[size];
        cudaMallocManaged(&data, size * sizeof(float));
        for (int i = 0; i < size; ++i) {
            data[i] = arr[i];
        }
    }

    // Overload the + operator to add two Array objects
    Array operator+(const Array &other) const {
        if (size != other.size) {
            std::cerr << "Error: Arrays must be of the same size to add." << std::endl;
            exit(EXIT_FAILURE);
        }

        Array result(*this);  // Create a copy of the current object

         // Allocate Unified Memory – accessible from CPU or GPU
        cudaMallocManaged(&result.data, size*sizeof(float));
        
        // Run kernel on 1M elements on the GPU
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        add<<<numBlocks, blockSize>>>(size, data, other.data, result.data);
        
        cudaDeviceSynchronize();

        return result;
    }

    Array dot(const Array &other) const {
        if (size != other.size) {
            std::cerr << "Error: Arrays must be of the same size to add." << std::endl;
            exit(EXIT_FAILURE);
        }

        Array result(*this);  // Create a copy of the current object

         // Allocate Unified Memory – accessible from CPU or GPU
        cudaMallocManaged(&result.data, size*sizeof(float));
        
        // Run kernel on 1M elements on the GPU
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        dot_arr<<<numBlocks, blockSize>>>(size, data, other.data, result.data);
        
        cudaDeviceSynchronize();

        return result;
    }

    // Function to print the array elements
    void printArray() const {
        for (int i = 0; i < size; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
};


int main() {
    float arr1[] = {1, 2, 3, 4, 5};
    float arr2[] = {5, 4, 3, 2, 1};

    int size = sizeof(arr1) / sizeof(arr1[0]);

    Array myArray1(arr1, size); 
    Array myArray2(arr2, size); 

    myArray1.printArray();
    myArray2.printArray();       

    Array add_result = myArray1 + myArray2;
    std::cout<<"Resulting array (add)"<<std::endl;
    add_result.printArray();

    Array dot_result = myArray1.dot(myArray2);
    std::cout<<"Resulting array (dot product)"<<std::endl;
    dot_result.printArray();

    return 0;
}
