#include <stdexcept>
#include "../../include/Core/Matrix2D.h"
#include "memory.h"

Matrix2D::Matrix2D(size_t rows, size_t cols, bool incremental) : rows(rows),
                                                                 cols(cols),
                                                                 transposed(false),
                                                                 incremental(incremental) {
    data.resize(rows * cols);
}

Matrix2D::Matrix2D(f32 value) : rows(1),
                                cols(1),
                                transposed(false),
                                incremental(false) {
    data.assign({value});
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

void Matrix2D::EachCellOperator(const Matrix2D &left, f32 (*functor)(f32), const Matrix2D *grad) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this)(i, j) += functor(left(i, j)) * (grad == nullptr ? 1.f : (*grad)(i, j));
            else
                (*this)(i, j) = functor(left(i, j)) * (grad == nullptr ? 1.f : (*grad)(i, j));
}

void Matrix2D::EachCellOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this)(i, j) += functor(left(i, j), right(i, j)) * (grad == nullptr ? 1.f : (*grad)(i, j));
            else
                (*this)(i, j) = functor(left(i, j), right(i, j)) * (grad == nullptr ? 1.f : (*grad)(i, j));
}

void Matrix2D::RowOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this)(i, j) += functor(left(i, j), right(0, j)) * (grad == nullptr ? 1.f : (*grad)(i, j));
            else
                (*this)(i, j) = functor(left(i, j), right(0, j)) * (grad == nullptr ? 1.f : (*grad)(i, j));
}

void Matrix2D::ColOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this)(i, j) += functor(left(i, j), right(i, 0)) * (grad == nullptr ? 1.f : (*grad)(i, j));
            else
                (*this)(i, j) = functor(left(i, j), right(i, 0)) * (grad == nullptr ? 1.f : (*grad)(i, j));
}

void Matrix2D::CellOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this)(i, j) += functor(left(i, j), right(0, 0)) * (grad == nullptr ? 1.f : (*grad)(i, j));
            else
                (*this)(i, j) = functor(left(i, j), right(0, 0)) * (grad == nullptr ? 1.f : (*grad)(i, j));
}


void Matrix2D::MergeRowOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
    for (size_t j = 0; j < cols; ++j) {
        f32 tmp = left(0, j);
        for (size_t i = 1; i < left.getRows(); ++i)
            tmp = functor(tmp, left(i, j));
        if (incremental)
            (*this)(0, j) += tmp * (grad == nullptr ? 1.f : (*grad)(0, j));
        else
            (*this)(0, j) = tmp * (grad == nullptr ? 1.f : (*grad)(0, j));
    }
}

void Matrix2D::MergeColOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
    for (size_t i = 0; i < rows; ++i) {
        f32 tmp = left(i, 0);
        for (size_t j = 1; j < left.getCols(); ++j)
            tmp = functor(tmp, left(i, j));
        if (incremental)
            (*this)(i, 0) += tmp * (grad == nullptr ? 1.f : (*grad)(i, 0));
        else
            (*this)(i, 0) = tmp * (grad == nullptr ? 1.f : (*grad)(i, 0));
    }
}

void Matrix2D::MergeCellOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
    f32 tmp = left(0, 0);
    for (size_t j = 1; j < left.getCols(); ++j)
        tmp = functor(tmp, left(0, j));

    for (size_t i = 1; i < left.getRows(); ++i)
        for (size_t j = 0; j < left.getCols(); ++j)
            tmp = functor(tmp, left(i, j));

    if (incremental)
        (*this)(0, 0) += tmp * (grad == nullptr ? 1.f : (*grad)(0, 0));
    else
        (*this)(0, 0) = tmp * (grad == nullptr ? 1.f : (*grad)(0, 0));
}

void Matrix2D::MultiplyOperator(const Matrix2D &left, const Matrix2D &right) {
    for (size_t i = 0; i < left.getRows(); ++i)
        for (size_t j = 0; j < right.getCols(); ++j) {
            f32 tmp = 0.f;
            for (size_t k = 0; k < left.getCols(); ++k)
                tmp += left(i, k) * right(k, j);
            if (incremental)
                (*this)(i, j) += tmp;
            else
                (*this)(i, j) = tmp;
        }
}

void Matrix2D::transpose() {
    transposed = !transposed;
}

void Matrix2D::Clean() {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            (*this)(i, j) = 0.f;
}

