#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <iomanip>
#include <stdexcept>

class Matrix {
    public:
        size_t rows, cols;

        Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
            data = new float[rows * cols]();
            std::fill(data, data + (rows*cols), 0.0f);
        }

        // Copy Constructor (Deep Copy)
        Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
            data = new float[rows * cols];
            std::copy(other.data, other.data + (rows * cols), data);
        }

        // Copy Assignment Operator (Deep Copy)
        Matrix& operator=(const Matrix& other) {
            if (this == &other) return *this; // Self-assignment check
            delete[] data;
            rows = other.rows;
            cols = other.cols;
            data = new float[rows * cols];
            std::copy(other.data, other.data + (rows * cols), data);
            return *this;
        }

        // Move Constructor (Efficient Transfer of Ownership)
        Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data) {
            other.data = nullptr;
            other.rows = other.cols = 0;
        }

        // Move Assignment Operator
        Matrix& operator=(Matrix&& other) noexcept {
            if (this == &other) return *this;
            delete[] data;
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            other.data = nullptr;
            other.rows = other.cols = 0;
            return *this;
        }

        // Destructor
        virtual ~Matrix() {
            delete[] data;
        }

        inline float& operator()(int i, int j) {
            return data[i * cols + j];
        }
        
        inline float operator()(int i, int j) const {
            return data[i * cols + j];
        }

        Matrix operator-() const {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result(i, j) = -result(i, j);
            return result;
        }

        inline float max() const {
            float maxval = data[0];
            for (int i = 0; i < rows*cols; i++)
                if (data[i] > maxval) maxval = data[i];
            return maxval;
        }

        inline float min() const {
            float minval = data[0];
            for (int i = 0; i < rows*cols; i++)
                if (data[i] < minval) minval = data[i];
            return minval;
        }

        void zero() {
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    data[i * cols + j] = 0.0f;
        }

        void print() const {
            for (int i = 0; i < rows; i++) {
                std::cout << "[ ";
                for (int j = 0; j < cols; j++) {
                    std::cout << std::setw(6) << data[i * cols + j] << std::setprecision(3) << " ";
                }
                std::cout << "]\n";
            }
        }
    
    private:
        float* data;

};

Matrix operator+(const Matrix A, const Matrix B) {
    if ((A.rows != B.rows) || (A.cols != B.cols))
        throw std::runtime_error("ERROR: Matrices must be the same shape to be added.");

    Matrix result(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < A.cols; j++)
            result(i, j) = A(i, j) + B(i, j);
    return result;
}

Matrix operator-(const Matrix A, const Matrix B) {
    if ((A.rows != B.rows) || (A.cols != B.cols))
        throw std::runtime_error("ERROR: Matrices must be the same shape to be subtracted.");

    Matrix result(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < A.cols; j++)
            result(i, j) = A(i, j) - B(i, j);
    return result;
}

Matrix operator*(const Matrix A, const Matrix B) {
    if ((A.rows != B.rows) || (A.cols != B.cols))
        throw std::runtime_error("ERROR: Matrices must be the same shape to be Hadamard multiplied.");

    Matrix result(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < A.cols; j++)
            result(i, j) = A(i, j) * B(i, j);
    return result;
}

Matrix operator*(const Matrix A, const float t) {
    Matrix result(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < A.cols; j++)
            result(i, j) = A(i, j) * t;
    return result;
}

Matrix operator*(const float t, const Matrix A) {
    return A * t;
}

Matrix operator-(const Matrix A, const float t) {
    Matrix result(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < A.cols; j++)
            result(i, j) = A(i, j) - t;
    return result;
}

Matrix operator-(const float t, const Matrix A) {
    return A - t;
}

Matrix operator/(const Matrix A, const float t) {
    return A * (1/t);
}

#endif