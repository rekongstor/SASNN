#include "../include/Matrix2D.h"
#include "memory.h"

Matrix2D::Matrix2D(size_t rows, size_t cols) : rows(rows),
                                               cols(cols) {
    data.resize(rows * cols);
}

f32 &Matrix2D::operator()(size_t row, size_t col) {
#if (DEBUG_LEVEL > 0)
    if (row > rows || col > cols)
        throw std::out_of_range("Matrix2D is out of range");
#endif
    return data[row * cols + col];
}

const f32 &Matrix2D::operator()(size_t row, size_t col) const {
#if (DEBUG_LEVEL > 0)
    if (row > rows || col > cols)
        throw std::out_of_range("Matrix2D is out of range");
#endif
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

void Matrix2D::CellOperator(const Matrix2D &left, f32 (*functor)(f32)) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            (*this)(i, j) = functor(left(i, j));
}

void Matrix2D::CellOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32)) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            (*this)(i, j) = functor(left(i, j), right(i, j));
}

void Matrix2D::RowOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32)) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            (*this)(i, j) = functor(left(i, j), right(0, j));
}

void Matrix2D::ColOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32)) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            (*this)(i, j) = functor(left(i, j), right(i, 0));
}

void Matrix2D::MergeRowOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32)) {
    for (size_t j = 0; j < cols; ++j) {
        f32 tmp = left(0,j);
        for (size_t i = 1; i < left.rows; ++i)
            tmp = functor(tmp, left(i, j));
        (*this)(0,j) = tmp;
    }
}

void Matrix2D::MergeColOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32)) {
    for (size_t i = 0; i < rows; ++i) {
        f32 tmp = left(i, 0);
        for (size_t j = 1; j < left.cols; ++j)
            tmp = functor(tmp, left(i, j));
        (*this)(i, 0) = tmp;
    }
}

void Matrix2D::MergeAllOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32)) {
    f32 tmp = left(0, 0);
    for (size_t j = 1; j < left.cols; ++j)
        tmp = functor(tmp, left(0, j));

    for (size_t i = 1; i < left.rows; ++i)
        for (size_t j = 0; j < left.cols; ++j)
            tmp = functor(tmp, left(i, j));

    (*this)(0,0) = tmp;
}
