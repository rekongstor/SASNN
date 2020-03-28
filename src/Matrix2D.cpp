#include "../include/Matrix2D.h"
#include "memory.h"

Matrix2D::Matrix2D(size_t rows, size_t cols) : rows(rows),
                                               cols(cols) {
    data.resize(rows * cols);
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

void Matrix2D::copyRow(size_t row, const f32 *data) {
    if (row > rows)
        throw std::out_of_range("Matrix2D is out of range");
    memcpy(this->data.data() + row * cols, data, sizeof(f32) * cols);
}

