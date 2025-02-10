#ifndef MATRIX_H
#define MATRIX_H

#ifndef tx
    #define tx 32
#endif

#ifndef ty
    #define ty 32
#endif

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#ifndef CHECK_CUDA
    #define CHECK_CUDA(call) check_cuda( (call), #call, __FILE__, __LINE__ )

    void check_cuda(cudaError_t result, char const *const func, const char * const file, int const line) {
        if (result) {
            std::cerr << "CUDA Error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "'\n";

            cudaDeviceReset();
            exit(99);
        }
    }
#endif

__global__ void subtract_matrix_kernel(const float* A, const float* B, float* result, int rows, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = row * cols + col;

    if ( (row < rows) && (col < cols) )
        result[idx] = A[idx] - B[idx]; 
}

__global__ void add_matrix_kernel(const float* A, const float* B, float* result, int rows, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = row * cols + col;

    if ( (row < rows) && (col < cols) )
        result[idx] = A[idx] + B[idx]; 
}

__global__ void hadamard_matrix_kernel(const float* A, const float* B, float* result, int rows, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = row * cols + col;

    if ( (row < rows) && (col < cols) )
        result[idx] = A[idx] * B[idx]; 
}

__global__ void matrix_times_const_kernel(const float* A, float t, float* result, int rows, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = row * cols + col;

    if ( (row < rows) && (col < cols) )
        result[idx] = A[idx] * t; 
}

__global__ void matrix_plus_const_kernel(const float* A, float t, float* result, int rows, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = row * cols + col;

    if ( (row < rows) && (col < cols) )
        result[idx] = A[idx] + t; 
}

__global__ void const_minus_matrix_kernel(const float* A, float t, float* result, int rows, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = row * cols + col;

    if ( (row < rows) && (col < cols) )
        result[idx] = t - A[idx]; 
}

__global__ void negate_matrix_kernel(const float* A, float* result, int rows, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = row * cols + col;

    if ( (row < rows) && (col < cols) )
        result[idx] = -A[idx]; 
}

class Matrix {
    public:
        float* data;  // Pointer to CUDA memory where matrix is stored
        size_t rows, cols;
        size_t n_bytes;

        Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), n_bytes(rows*cols*sizeof(float)) {
            CHECK_CUDA(cudaMalloc((void**)&data, n_bytes)); // Allocate memory on GPU
            CHECK_CUDA(cudaMemset(data, 0, n_bytes)); // Set all entries to zero 
        }

        // Initialization of Vector (N x 1 matrix) from list
        Matrix(std::initializer_list<float> list) : rows(1), cols(list.size()), n_bytes(rows*cols*sizeof(float)) {
            CHECK_CUDA(cudaMalloc((void**)&data, n_bytes));
            CHECK_CUDA(cudaMemcpy(data, list.begin(), n_bytes, cudaMemcpyHostToDevice));
        }

        // Copy Constructor (Deep Copy)
        Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), n_bytes(other.rows*other.cols*sizeof(float)) {
            // Allocate data on the GPU for the copy
            CHECK_CUDA(cudaMalloc((void**)&data, n_bytes));

            // Copy data from other into this
            CHECK_CUDA(cudaMemcpy(data, other.data, n_bytes, cudaMemcpyDeviceToDevice));
        }

        // Copy Assignment Operator (Deep Copy)
        Matrix& operator=(const Matrix& other) {
            // Do nothing if self-assignment
            if (this == &other) return *this;
            
            // Free the old memory (this->data)
            CHECK_CUDA(cudaFree(data));
            
            // Copy other matrix into this one
            rows = other.rows;
            cols = other.cols;
            n_bytes = other.n_bytes;
            
            CHECK_CUDA(cudaMalloc((void**)&data, n_bytes));
            CHECK_CUDA(cudaMemcpy(data, other.data, n_bytes, cudaMemcpyDeviceToDevice));
            return *this;
        }

        // Move Constructor (Efficient Transfer of Ownership)
        // Move constructor (efficiently transfers ownership)
        Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data), n_bytes(other.n_bytes) {
            other.data = nullptr;  // Nullify old pointer
        }

        // Move Assignment Operator
        Matrix& operator=(Matrix&& other) noexcept {
            // Do nothing if self-assignment
            if (this == &other) return *this;
            
            // Free this->data
            CHECK_CUDA(cudaFree(data));

            // Transfer ownership of the data ptr to this object
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            n_bytes = other.n_bytes;
            other.data = nullptr;
            other.rows = other.cols = 0;
            return *this;
        }

        // Destructor
        virtual ~Matrix() {
            CHECK_CUDA(cudaFree(data));
        }

        Matrix operator-() const {
            Matrix result(rows, cols);
            
            int tx_ = (cols < tx) ? cols : tx;
            int ty_ = (rows < ty) ? rows : ty;
            dim3 threads(tx_, ty_);
            dim3 blocks(cols / tx_ + 1, rows / ty_ + 1);;

            negate_matrix_kernel
                <<<blocks, threads>>>
                (this->data, result.data, rows, cols);
            
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError());
            return result;
        }

        inline float max() const {
            thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(data);
            return *thrust::max_element(dev_ptr, dev_ptr + rows*cols);
        }

        inline float min() const {
            thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(data);
            return *thrust::min_element(dev_ptr, dev_ptr + rows*cols);
        }

        void zero() {
            CHECK_CUDA(cudaMemset(data, 0, n_bytes));
        }

        void print() const {
            float* h_data = new float[rows*cols]();
            CHECK_CUDA(cudaMemcpy(h_data, data, n_bytes, cudaMemcpyDeviceToHost));

            for (int i = 0; i < rows; i++) {
                std::cout << "[ ";
                for (int j = 0; j < cols; j++) {
                    std::cout << std::setw(6) << h_data[i * cols + j] << std::setprecision(3) << " ";
                }
                std::cout << "]\n";
            }

            delete[] h_data;
        }
    
};

Matrix operator+(const Matrix& A, const Matrix& B) {
    if ((A.rows != B.rows) || (A.cols != B.cols))
        throw std::runtime_error("ERROR: Matrices must be the same shape to be added.");

    Matrix result(A.rows, A.cols);

    int tx_ = (A.cols < tx) ? A.cols : tx;
    int ty_ = (A.rows < ty) ? A.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(A.cols/tx_ + 1, A.rows/ty_ + 1);
    add_matrix_kernel<<<blocks,threads>>>(A.data, B.data, result.data, A.rows, A.cols);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    return result;
}

Matrix operator-(const Matrix& A, const Matrix& B) {
    if ((A.rows != B.rows) || (A.cols != B.cols))
        throw std::runtime_error("ERROR: Matrices must be the same shape to be subtracted.");

    Matrix result(A.rows, A.cols);
    
    int tx_ = (A.cols < tx) ? A.cols : tx;
    int ty_ = (A.rows < ty) ? A.rows : ty;
    
    dim3 threads(tx_, ty_);
    dim3 blocks(A.cols/tx_ + 1, A.rows/ty_ + 1);
    subtract_matrix_kernel<<<blocks,threads>>>(A.data, B.data, result.data, A.rows, A.cols);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    return result;
}

Matrix operator*(const Matrix& A, const Matrix& B) {
    if ((A.rows != B.rows) || (A.cols != B.cols))
        throw std::runtime_error("ERROR: Matrices must be the same shape to be Hadamard multiplied.");

    Matrix result(A.rows, A.cols);

    int tx_ = (A.cols < tx) ? A.cols : tx;
    int ty_ = (A.rows < ty) ? A.rows : ty;

    dim3 threads(tx_, ty_);
    dim3 blocks(A.cols/tx_ + 1, A.rows/ty_ + 1);
    hadamard_matrix_kernel<<<blocks,threads>>>(A.data, B.data, result.data, A.rows, A.cols);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    return result;
}

Matrix operator*(const Matrix& A, const float t) {
    Matrix result(A.rows, A.cols);

    int tx_ = (A.cols < tx) ? A.cols : tx;
    int ty_ = (A.rows < ty) ? A.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(A.cols/tx_ + 1, A.rows/ty_ + 1);
    matrix_times_const_kernel<<<blocks,threads>>>(A.data, t, result.data, A.rows, A.cols);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    return result;
}

Matrix operator*(const float t, const Matrix& A) {
    return A * t;
}

Matrix operator-(const Matrix& A, const float t) {
    Matrix result(A.rows, A.cols);

    int tx_ = (A.cols < tx) ? A.cols : tx;
    int ty_ = (A.rows < ty) ? A.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(A.cols/tx_ + 1, A.rows/ty_ + 1);
    matrix_plus_const_kernel<<<blocks,threads>>>(A.data, -t, result.data, A.rows, A.cols);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    return result;
}

Matrix operator-(const float t, const Matrix& A) {
    Matrix result(A.rows, A.cols);
    
    int tx_ = (A.cols < tx) ? A.cols : tx;
    int ty_ = (A.rows < ty) ? A.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(A.cols/tx_ + 1, A.rows/ty_ + 1);
    const_minus_matrix_kernel<<<blocks,threads>>>(A.data, t, result.data, A.rows, A.cols);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    return result;
}

Matrix operator/(const Matrix& A, const float t) {
    if (t == 0.0f) throw std::runtime_error("ERROR: Division by zero.");
   
    return A * (1/t);
}

#endif