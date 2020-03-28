#include "../include/Matrix2D.h"
#include "memory.h"

Matrix2D::Matrix2D(size_t rows, size_t cols) : rows(rows), cols(cols) {
    data = new f32[rows * cols];
}

Matrix2D::~Matrix2D() {
    delete[] data;
}

Matrix2D::Matrix2D(f32 *data, size_t rows, size_t cols) : rows(rows), cols(cols), data(nullptr) {
    data = new f32[rows * cols];
    memcpy(this->data, data, sizeof(f32) * rows * cols);
}

f32 &Matrix2D::operator()(size_t row, size_t col) {
    if (row > rows || col > cols)
        throw std::out_of_range("Matrix2D is out of range");
    return data[row * cols + col];
}

size_t Matrix2D::getRows() const {
    return rows;
}

size_t Matrix2D::getCols() const {
    return cols;
}

