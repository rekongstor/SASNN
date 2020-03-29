#pragma once

#include <vector>
#include "stdafx.h"

class Matrix2D {
    size_t rows;
    size_t cols;
    bool transposed;
    bool incremental;
    std::vector<f32> data;
public:
    Matrix2D(size_t rows, size_t cols, bool incremental = false);
    f32 &operator()(size_t row, size_t col);
    const f32 &operator()(size_t row, size_t col) const;

    [[nodiscard]] size_t getRows() const;
    [[nodiscard]] size_t getCols() const;
    void copyRow(size_t row, const f32 *data);
    void transpose();
    void Clean();

    /**
     * Performs unary functor on the corresponding cell
     * @param left : [rows x cols]
     */
    void CellOperator(const Matrix2D &left, f32 (*functor)(f32));
    /**
     * Performs binary functor on the corresponding cell
     * @param left : [rows x cols]
     * @param right : [rows x cols]
     */
    void CellOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32));
    /**
     * Performs unary functor on the each row
     * @param left [rows x cols]
     * @param right [1 x cols]
     */
    void RowOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32));
    /**
     * Performs unary functor on the each column
     * @param left [rows x cols]
     * @param right [rows x 1]
     */
    void ColOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32));
    /**
     * Merges all rows into one according to the binary functor
     * @param left [ANY x cols]
     */
    void MergeRowOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32));
    /**
     * Merges all columns into one according to the binary functor
     * @param left [rows x ANY]
     */
    void MergeColOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32));
    /**
     * Merges everything into one according to the binary functor (increasing columns for each row)
     * @param left [ANY x ANY]
     */
    void MergeAllOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32));
};

