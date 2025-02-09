#ifndef VECTORS_H
#define VECTORS_H

#include <iostream>
#include <iomanip>

class Vector {
    public:
        size_t dim;

        Vector(size_t dim) : dim(dim) { 
            data = new float[dim]; 
            std::fill(data, data + dim, 0.0f);
        }

        // Copy Constructor (Deep Copy)
        Vector(const Vector& other) : dim(other.dim) {
            data = new float[dim];
            std::copy(other.data, other.data + dim, data);
        }

        // Copy Assignment Operator (Deep Copy)
        Vector& operator=(const Vector& other) {
            if (this == &other) return *this; // Self-assignment check
            delete[] data;
            dim = other.dim;
            data = new float[dim];
            std::copy(other.data, other.data + dim, data);
            return *this;
        }

        // Move Constructor (Efficient Transfer of Ownership)
        Vector(Vector&& other) noexcept : dim(other.dim), data(other.data) {
            other.data = nullptr;
            other.dim = 0;
        }

        // Move Assignment Operator
        Vector& operator=(Vector&& other) noexcept {
            if (this == &other) return *this;
            delete[] data;
            dim = other.dim;
            data = other.data;
            other.data = nullptr;
            other.dim = 0;
            return *this;
        }

        virtual ~Vector() { 
            delete[] data;
        }

        inline float& operator() (const int i) {
            return data[i];
        }

        inline float operator() (const int i) const {
            return data[i];
        }

        Vector operator-() const {
            Vector result(dim);
            for (int i = 0; i < dim; i++)
                result(i) = -result(i);
            return result;
        }

        inline void linspace(const float start, const float end) {
            float delta = (end - start) / (dim - 1);
            for (int i = 0; i < dim; i++) {
                *(data + i) = start + i * delta;
            }
        }

        void print() {
            std::cout << "[ ";
            for (int i = 0; i < dim; i++) {
                std::cout << std::setw(6) << data[i] << std::setprecision(2) << " ";
            }
            std::cout << "]";
        }
    
    private:
        float* data;
};

Vector operator+(const Vector A, const Vector B) {
    if (A.dim != B.dim)
        throw std::runtime_error("ERROR: Vectors must be the same shape to be added.");

    Vector result(A.dim);
    for (int i = 0; i < A.dim; i++)
            result(i) = A(i) + B(i);
    return result;
}

Vector operator*(const Vector A, const Vector B) {
    if (A.dim != B.dim)
        throw std::runtime_error("ERROR: Vectors must be the same shape to be Hadamard multiplied.");

    Vector result(A.dim);
    for (int i = 0; i < A.dim; i++)
            result(i) = A(i) * B(i);
    return result;
}

Vector operator*(const Vector A, const float t) {
    Vector result(A.dim);
    for (int i = 0; i < A.dim; i++)
            result(i) = A(i) * t;
    return result;
}

Vector operator*(const float t, const Vector A) {
    return A * t;
}

#endif