#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "matrix.h"

// Meshgrid
__global__ void meshgrid_kernel(const float* x_1D, const float* y_1D, float* X_2D, float* Y_2D, int num_x, int num_y) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row >= num_x) return;
    if (col >= num_y) return;

    X_2D[row * num_y + col] = x_1D[row];
    Y_2D[row * num_y + col] = y_1D[col];
}
void meshgrid(const Matrix& x_1D, const Matrix& y_1D, Matrix& X_2D, Matrix& Y_2D) {
    int tx_ = (X_2D.cols < tx) ? X_2D.cols : tx;
    int ty_ = (X_2D.rows < ty) ? X_2D.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(X_2D.cols/tx_ + 1, X_2D.rows/ty_ + 1);

    meshgrid_kernel<<<blocks, threads>>>(x_1D.data, y_1D.data, X_2D.data, Y_2D.data, X_2D.rows, X_2D.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

// Central Difference - X
__global__ void interior_central_difference_x_uniform_kernel(const float* F, float dx, float* diff, int rows, int cols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (
        (row >= 1) 
        && (row < rows - 1)
        && (col >= 1)
        && (col < cols - 1)
    ) {
        int center_idx   = (row + 0) * cols + col;
        int forward_idx  = (row + 1) * cols + col;
        int backward_idx = (row - 1) * cols + col;

        diff[center_idx] = (F[forward_idx] - F[backward_idx]) / (2 * dx);
    }
}
void interior_central_difference_x_uniform(const Matrix& F, const float dx, Matrix& diff) {
    int tx_ = (F.cols < tx) ? F.cols : tx;
    int ty_ = (F.rows < ty) ? F.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(F.cols/tx_ + 1, F.rows/ty_ + 1);

    interior_central_difference_x_uniform_kernel<<<blocks, threads>>>(F.data, dx, diff.data, F.rows, F.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

// Central Difference - Y
__global__ void interior_central_difference_y_uniform_kernel(const float* F, float dx, float* diff, int rows, int cols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (
        (row >= 1) 
        && (row < rows - 1)
        && (col >= 1)
        && (col < cols - 1)
    ) {
        int center_idx   = row * cols + (col + 0);
        int forward_idx  = row * cols + (col + 1);
        int backward_idx = row * cols + (col - 1);

        diff[center_idx] = (F[forward_idx] - F[backward_idx]) / (2 * dx);
    }
}
void interior_central_difference_y_uniform(const Matrix& F, const float dy, Matrix& diff) {
    int tx_ = (F.cols < tx) ? F.cols : tx;
    int ty_ = (F.rows < ty) ? F.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(F.cols/tx_ + 1, F.rows/ty_ + 1);

    interior_central_difference_y_uniform_kernel<<<blocks, threads>>>(F.data, dy, diff.data, F.rows, F.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

// Laplacian - 5 Point Stencil
__global__ void interior_laplace_5point_stencil_uniform_kernel(const float* F, float d, float* diff, int rows, int cols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (
        (row >= 1) 
        && (row < rows - 1)
        && (col >= 1)
        && (col < cols - 1)
    ) {
        int center_idx = (row + 0) * cols + (col + 0);
        int right_idx  = (row + 1) * cols + (col + 0);
        int left_idx   = (row - 1) * cols + (col + 0);
        int up_idx     = (row + 0) * cols + (col + 1);
        int down_idx   = (row + 0) * cols + (col - 1);

        diff[center_idx] = (
            F[right_idx] + F[up_idx] + F[left_idx] + F[down_idx]
            - 4 * F[center_idx]
        ) / (d * d);
    }
}
void interior_laplace_5point_stencil_uniform(const Matrix& F, const float d, Matrix& diff) {
    int tx_ = (F.cols < tx) ? F.cols : tx;
    int ty_ = (F.rows < ty) ? F.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(F.cols/tx_ + 1, F.rows/ty_ + 1);

    interior_laplace_5point_stencil_uniform_kernel<<<blocks, threads>>>(F.data, d, diff.data, F.rows, F.cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

// Pressure Poisson Jacobi Iteration
__global__ void pressure_poisson_jacobi_iteration_kernel(const float* p_prev, const float* rhs, float dx, float* p_next, int rows, int cols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if indices are too small
    if (row < 1) return;
    if (col < 1) return;

    // Check if indices are too large
    if (row >= rows - 1) return;
    if (col >= cols - 1) return;

    int center_idx = (row + 0) * cols + (col + 0);
    int left_idx   = (row - 1) * cols + (col + 0);
    int right_idx  = (row + 1) * cols + (col + 0);
    int up_idx     = (row + 0) * cols + (col + 1);
    int down_idx   = (row + 0) * cols + (col - 1);

    float A = p_prev[down_idx] + p_prev[up_idx]
            + p_prev[left_idx] + p_prev[right_idx];
    float B = dx * dx * rhs[center_idx];
    float C = (A - B) / 4;
    
    p_next[center_idx] = C;
}
void pressure_poisson_jacobi_iteration(Matrix& p_prev, Matrix& rhs, float dx, Matrix& p_next) {
    int tx_ = (p_prev.cols < tx) ? p_prev.cols : tx;
    int ty_ = (p_prev.rows < ty) ? p_prev.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(p_prev.cols/tx_ + 1, p_prev.rows/ty_ + 1);

    pressure_poisson_jacobi_iteration_kernel<<<blocks, threads>>>(
        p_prev.data, rhs.data, dx, p_next.data, p_prev.rows, p_prev.cols
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

// High-accuracy derivatives for smoke advection
// X-Direction Third-Order Difference
__global__ void interior_third_order_difference_x_kernel(const float* F, float dx, float* diff, int rows, int cols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row >= rows || col >= cols) return;

    int center_idx = row * cols + col;

    // First two rows - Forward difference
    if (row < 2) {
        // f'(x) ≈ (-11f(x) + 18f(x+h) - 9f(x+2h) + 2f(x+3h))/(6h)
        int f0_idx = row * cols + col;
        int f1_idx = (row + 1) * cols + col;
        int f2_idx = (row + 2) * cols + col;
        int f3_idx = (row + 3) * cols + col;

        diff[center_idx] = (
            -11.0f * F[f0_idx]
            + 18.0f * F[f1_idx]
            - 9.0f * F[f2_idx]
            + 2.0f * F[f3_idx]
        ) / (6.0f * dx);
    }
    // Last two rows - Backward difference
    else if (row >= rows - 2) {
        // f'(x) ≈ (11f(x) - 18f(x-h) + 9f(x-2h) - 2f(x-3h))/(6h)
        int f0_idx = row * cols + col;
        int f1_idx = (row - 1) * cols + col;
        int f2_idx = (row - 2) * cols + col;
        int f3_idx = (row - 3) * cols + col;

        diff[center_idx] = (
            11.0f * F[f0_idx]
            - 18.0f * F[f1_idx]
            + 9.0f * F[f2_idx]
            - 2.0f * F[f3_idx]
        ) / (6.0f * dx);
    }
    // Interior points - Central difference
    else {
        // f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h))/(12h)
        int forward2_idx = (row + 2) * cols + col;
        int forward1_idx = (row + 1) * cols + col;
        int backward1_idx = (row - 1) * cols + col;
        int backward2_idx = (row - 2) * cols + col;

        diff[center_idx] = (
            -F[forward2_idx]
            + 8.0f * F[forward1_idx]
            - 8.0f * F[backward1_idx]
            + F[backward2_idx]
        ) / (12.0f * dx);
    }
}

// Y-Direction Third-Order Difference
__global__ void interior_third_order_difference_y_kernel(const float* F, float dy, float* diff, int rows, int cols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row >= rows || col >= cols) return;

    int center_idx = row * cols + col;

    // First two columns - Forward difference
    if (col < 2) {
        // f'(y) ≈ (-11f(y) + 18f(y+h) - 9f(y+2h) + 2f(y+3h))/(6h)
        int f0_idx = row * cols + col;
        int f1_idx = row * cols + (col + 1);
        int f2_idx = row * cols + (col + 2);
        int f3_idx = row * cols + (col + 3);

        diff[center_idx] = (
            -11.0f * F[f0_idx]
            + 18.0f * F[f1_idx]
            - 9.0f * F[f2_idx]
            + 2.0f * F[f3_idx]
        ) / (6.0f * dy);
    }
    // Last two columns - Backward difference
    else if (col >= cols - 2) {
        // f'(y) ≈ (11f(y) - 18f(y-h) + 9f(y-2h) - 2f(y-3h))/(6h)
        int f0_idx = row * cols + col;
        int f1_idx = row * cols + (col - 1);
        int f2_idx = row * cols + (col - 2);
        int f3_idx = row * cols + (col - 3);

        diff[center_idx] = (
            11.0f * F[f0_idx]
            - 18.0f * F[f1_idx]
            + 9.0f * F[f2_idx]
            - 2.0f * F[f3_idx]
        ) / (6.0f * dy);
    }
    // Interior points - Central difference
    else {
        // f'(y) ≈ (-f(y+2h) + 8f(y+h) - 8f(y-h) + f(y-2h))/(12h)
        int forward2_idx = row * cols + (col + 2);
        int forward1_idx = row * cols + (col + 1);
        int backward1_idx = row * cols + (col - 1);
        int backward2_idx = row * cols + (col - 2);

        diff[center_idx] = (
            -F[forward2_idx]
            + 8.0f * F[forward1_idx]
            - 8.0f * F[backward1_idx]
            + F[backward2_idx]
        ) / (12.0f * dy);
    }
}

// Wrapper functions remain the same
void interior_third_order_difference_x(const Matrix& F, const float dx, Matrix& diff) {
    int tx_ = (F.cols < tx) ? F.cols : tx;
    int ty_ = (F.rows < ty) ? F.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(F.cols/tx_ + 1, F.rows/ty_ + 1);

    interior_third_order_difference_x_kernel<<<blocks, threads>>>(
        F.data, dx, diff.data, F.rows, F.cols
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void interior_third_order_difference_y(const Matrix& F, const float dy, Matrix& diff) {
    int tx_ = (F.cols < tx) ? F.cols : tx;
    int ty_ = (F.rows < ty) ? F.rows : ty;
    dim3 threads(tx_, ty_);
    dim3 blocks(F.cols/tx_ + 1, F.rows/ty_ + 1);

    interior_third_order_difference_y_kernel<<<blocks, threads>>>(
        F.data, dy, diff.data, F.rows, F.cols
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

#endif