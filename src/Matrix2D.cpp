#include "../include/Matrix2D.h"
#include "memory.h"

Matrix2D::Matrix2D(size_t rows, size_t cols, bool incremental) : rows(rows),
                                                                 cols(cols),
                                                                 transposed(false),
                                                                 incremental(incremental) {
    data.resize(rows * cols);
}

f32 &Matrix2D::operator()(size_t row, size_t col) {
    if (!transposed) {
#if (DEBUG_LEVEL > 0)
        if (row >= rows || col >= cols)
            throw std::out_of_range("Matrix2D is out of range");
#endif
        return data[row * cols + col];
    }
#if (DEBUG_LEVEL > 0)
    if (col >= rows || row >= cols)
        throw std::out_of_range("Matrix2D is out of range");
#endif
    return data[col * cols + row];
}

const f32 &Matrix2D::operator()(size_t row, size_t col) const {
    if (!transposed) {
#if (DEBUG_LEVEL > 0)
        if (row >= rows || col >= cols)
            throw std::out_of_range("Matrix2D is out of range");
#endif
        return data[row * cols + col];
    }
#if (DEBUG_LEVEL > 0)
    if (col >= rows || row >= cols)
        throw std::out_of_range("Matrix2D is out of range");
#endif
    return data[col * cols + row];
}

size_t Matrix2D::getRows() const {
    if (transposed)
        return cols;
    return rows;
}

size_t Matrix2D::getCols() const {
    if (transposed)
        return rows;
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
            if (incremental)
                (*this)(i, j) += functor(left(i, j));
            else
                (*this)(i, j) = functor(left(i, j));
}

void Matrix2D::CellOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32)) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this)(i, j) += functor(left(i, j), right(i, j));
            else
                (*this)(i, j) = functor(left(i, j), right(i, j));
}

void Matrix2D::RowOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32)) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this)(i, j) += functor(left(i, j), right(0, j));
            else
                (*this)(i, j) = functor(left(i, j), right(0, j));
}

void Matrix2D::ColOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32)) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this)(i, j) += functor(left(i, j), right(i, 0));
            else
                (*this)(i, j) = functor(left(i, j), right(i, 0));
}

void Matrix2D::MergeRowOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32)) {
    for (size_t j = 0; j < cols; ++j) {
        f32 tmp = left(0, j);
        for (size_t i = 1; i < left.rows; ++i)
            tmp = functor(tmp, left(i, j));
        if (incremental)
            (*this)(0, j) += tmp;
        else
            (*this)(0, j) = tmp;
    }
}

void Matrix2D::MergeColOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32)) {
    for (size_t i = 0; i < rows; ++i) {
        f32 tmp = left(i, 0);
        for (size_t j = 1; j < left.cols; ++j)
            tmp = functor(tmp, left(i, j));
        if (incremental)
            (*this)(i, 0) += tmp;
        else
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

    if (incremental)
        (*this)(0, 0) += tmp;
    else
        (*this)(0, 0) = tmp;
}

void Matrix2D::transpose() {
    transposed = !transposed;
}

void Matrix2D::Clean() {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            (*this)(i, j) = 0.f;
}
