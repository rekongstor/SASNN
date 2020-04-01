#include <stdexcept>
#include <cmath>
#include <iostream>
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


const f32 &Matrix2D::operator()(size_t row, size_t col) const {
    if (!transposed) {
        row %= rows;
        col %= cols;
        return data[row * cols + col];
    }
    row %= cols;
    col %= rows;
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

void Matrix2D::EachCellOperator(const Matrix2D &left, f32 (*functor)(f32), const Matrix2D *grad) {
#pragma omp parallel for
    for (s64 i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this).setCell(i, j, (*this)(i, j) + functor(left(i, j)) * (grad == nullptr ? 1.f : (*grad)(i, j)));
            else
                (*this).setCell(i, j, functor(left(i, j)) * (grad == nullptr ? 1.f : (*grad)(i, j)));
}

void Matrix2D::EachCellOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
#pragma omp parallel for
    for (s64 i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this).setCell(i, j, (*this)(i, j) + functor(left(i, j), right(i, j)) * (grad == nullptr ? 1.f : (*grad)(i, j)));
            else
                (*this).setCell(i, j, functor(left(i, j), right(i, j)) * (grad == nullptr ? 1.f : (*grad)(i, j)));
}

void Matrix2D::RowOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
#pragma omp parallel for
    for (s64 i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this).setCell(i, j, (*this)(i, j) + functor(left(i, j), right(0, j)) * (grad == nullptr ? 1.f : (*grad)(i, j)));
            else
                (*this).setCell(i, j, functor(left(i, j), right(0, j)) * (grad == nullptr ? 1.f : (*grad)(i, j)));
}

void Matrix2D::ColOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
#pragma omp parallel for
    for (s64 i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this).setCell(i, j, (*this)(i, j) + functor(left(i, j), right(i, 0)) * (grad == nullptr ? 1.f : (*grad)(i, j)));
            else
                (*this).setCell(i, j, functor(left(i, j), right(i, 0)) * (grad == nullptr ? 1.f : (*grad)(i, j)));
}

void Matrix2D::CellOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad) {
#pragma omp parallel for
    for (s64 i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (incremental)
                (*this).setCell(i, j, (*this)(i, j) + functor(left(i, j), right(0, 0)) * (grad == nullptr ? 1.f : (*grad)(i, j)));
            else
                (*this).setCell(i, j, functor(left(i, j), right(0, 0)) * (grad == nullptr ? 1.f : (*grad)(i, j)));
}


void Matrix2D::MergeRowsOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32), const Matrix2D *grad, f32(*initFunctor)(const f32)) {
#pragma omp parallel for
    for (s64 j = 0; j < cols; ++j) {
        f32 tmp;
        if (initFunctor == nullptr)
            tmp = left(0, j);
        else
            tmp = initFunctor(left(0, j));

        for (size_t i = 1; i < left.getRows(); ++i)
            tmp = functor(tmp, left(i, j));
        if (incremental)
            (*this).setCell(0, j, (*this)(0, j) + tmp * (grad == nullptr ? 1.f : (*grad)(0, j)));
        else
            (*this).setCell(0, j, tmp * (grad == nullptr ? 1.f : (*grad)(0, j)));
    }
}

void Matrix2D::MergeColsOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32), const Matrix2D *grad, f32(*initFunctor)(const f32)) {
#pragma omp parallel for
    for (s64 i = 0; i < rows; ++i) {
        f32 tmp;
        if (initFunctor == nullptr)
            tmp = left(i, 0);
        else
            tmp = initFunctor(left(i, 0));

        for (size_t j = 1; j < left.getCols(); ++j)
            tmp = functor(tmp, left(i, j));
        if (incremental)
            (*this).setCell(i, 0, (*this)(i, 0) + tmp * (grad == nullptr ? 1.f : (*grad)(i, 0)));
        else
            (*this).setCell(i, 0, tmp * (grad == nullptr ? 1.f : (*grad)(i, 0)));
    }
}

void Matrix2D::MergeCellsOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32), const Matrix2D *grad, f32(*initFunctor)(const f32)) {
    f32 tmp;
    if (initFunctor == nullptr)
        tmp = left(0, 0);
    else
        tmp = initFunctor(left(0, 0));

    for (size_t j = 1; j < left.getCols(); ++j)
        tmp = functor(tmp, left(0, j));

#pragma omp parallel for
    for (s64 i = 1; i < left.getRows(); ++i)
        for (size_t j = 0; j < left.getCols(); ++j)
            tmp = functor(tmp, left(i, j));

    if (incremental)
        (*this).setCell(0, 0, (*this)(0, 0) + tmp * (grad == nullptr ? 1.f : (*grad)(0, 0)));
    else
        (*this).setCell(0, 0, tmp * (grad == nullptr ? 1.f : (*grad)(0, 0)));
}

void Matrix2D::MultiplyOperator(const Matrix2D &left, const Matrix2D &right) {
#pragma omp parallel for
    for (s64 i = 0; i < left.getRows(); ++i)
        for (size_t j = 0; j < right.getCols(); ++j) {
            f32 tmp = 0.f;
            for (size_t k = 0; k < left.getCols(); ++k)
                tmp += left(i, k) * right(k, j);
            if (incremental)
                (*this).setCell(i, j, (*this)(i, j) + tmp);
            else
                (*this).setCell(i, j, tmp);
        }
}


void Matrix2D::Transpose() {
    transposed = !transposed;
}

void Matrix2D::Clean() {
#pragma omp parallel for
    for (s64 i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            (*this).setCell(i, j, 0.f);
}

void Matrix2D::Fill(f32 value) {
#pragma omp parallel for
    for (s64 i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            (*this).setCell(i, j, value);
}

void Matrix2D::setCell(size_t row, size_t col, f32 val) {
    if (std::isinf(val))
        if ((val) > 0.f)
            val = std::numeric_limits<f32>::max();
        else
            val = std::numeric_limits<f32>::min();
    if (std::isnan(val))
        if ((val) > 0.f)
            val = 0.f;
        else
            val = -0.f;

    if (!transposed) {
#if (DEBUG_LEVEL > 0)
        if (row >= rows || col >= cols)
            throw std::out_of_range("Matrix2D is out of range");
#endif
        data[row * cols + col] = val;
        return;
    }
#if (DEBUG_LEVEL > 0)
    if (col >= rows || row >= cols)
        throw std::out_of_range("Matrix2D is out of range");
#endif
    data[col * cols + row] = val;
}

void Matrix2D::Print() const {
    for (size_t i = 0; i < getRows(); ++i) {
        for (size_t j = 0; j < getCols(); ++j)
            printf("%.3f ", operator()(i, j));
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Matrix2D::AssignData(f32 *src) {
    memcpy(data.data(), src, rows * cols * sizeof(f32));
}


void Matrix2D::FindColOperator(const Matrix2D &left, f32 neutralValue, bool (*functor)(const f32, const f32), const Matrix2D *grad) {
    f32 found_value;
    size_t found_index;
#pragma omp parallel for
    for (s64 j = 0; j < getCols(); ++j) {
        found_value = neutralValue;
        found_index = -1;
        for (size_t i = 0; i < getRows(); ++i)
            if (functor(left(i, j), found_value)) {
                found_value = left(i, j);
                found_index = i;
            }
        for (size_t i = 0; i < getRows(); ++i)
            setCell(i, j, (grad == nullptr ? 1.f : (*grad)(i, j)) * (i == found_index ? 1.f : 0.f));
    }
}

void Matrix2D::FindRowOperator(const Matrix2D &left, f32 neutralValue, bool (*functor)(const f32, const f32), const Matrix2D *grad) {
    f32 found_value;
    size_t found_index;
#pragma omp parallel for
    for (s64 i = 0; i < getRows(); ++i) {
        found_value = neutralValue;
        found_index = -1;
        for (size_t j = 0; j < getCols(); ++j)
            if (functor(left(i, j), found_value)) {
                found_value = left(i, j);
                found_index = j;
            }
        for (size_t j = 0; j < getCols(); ++j)
            setCell(i, j, (grad == nullptr ? 1.f : (*grad)(i, j)) * (j == found_index ? 1.f : 0.f));
    }

}
