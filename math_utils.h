#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "vectors.h"
#include "matrix.h"

// Meshgrid
void meshgrid(const Vector& A, const Vector& B, Matrix& retA, Matrix& retB) {
    for (int i = 0; i < A.dim; i++) {
        for (int j = 0; j < B.dim; j++) {
            retA(i, j) = A(i);
            retB(i, j) = B(j);
        }
    }
}

// Derivative operators
void interior_central_difference_x_uniform(const Matrix& F, const float dx, Matrix& diff) {
    for (int i = 1; i < diff.rows - 1; i++) {
        for (int j = 1; j < diff.cols - 1; j++) {
            diff(i, j) = (F(i + 1, j) - F(i - 1, j)) / (2 * dx);
        }
    }
}

void interior_central_difference_y_uniform(const Matrix& F, const float dy, Matrix& diff) {
    for (int i = 1; i < diff.rows - 1; i++) {
        for (int j = 1; j < diff.cols - 1; j++) {
            diff(i, j) = (F(i, j + 1) - F(i, j - 1)) / (2 * dy);
        }
    }
}

void interior_laplace_5point_stencil_uniform(const Matrix& F, const float d, Matrix& diff) {
    /*
                      F(i, j + 1)
                           |
    F(i - 1, j) -----   F(i, j)   ----- F(i + 1, j)
                           |
                      F(i, j - 1)
    */
    for (int i = 1; i < diff.rows - 1; i++) {
        for (int j = 1; j < diff.cols - 1; j++) {
            diff(i, j) = (
                F(i + 1, j) + F(i, j + 1)
                + F(i - 1, j) + F(i, j - 1)
                - 4 * F(i, j)
            ) / (d*d);
        }
    }
}

#endif