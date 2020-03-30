#pragma once

#include <vector>
#include "../../include/stdafx.h"

class Matrix2D {
    size_t rows;
    size_t cols;
    bool transposed;
    bool incremental;
    std::vector<f32> data;
public:
    Matrix2D(size_t rows, size_t cols, bool incremental = false);
    explicit Matrix2D(f32 value);
    void setCell(size_t row, size_t col, f32 val);
    const f32 &operator()(size_t row, size_t col) const;

    [[nodiscard]] size_t getRows() const;
    [[nodiscard]] size_t getCols() const;
    void CopyRow(size_t row, const f32 *data);
    void Transpose();
    void Clean();
    void Print() const;
    void Fill(f32 value);

    /**
     * Performs unary functor on the corresponding cell
     * @param left : [rows x cols]
     */
    void EachCellOperator(const Matrix2D &left, f32 (*functor)(f32), const Matrix2D *grad = nullptr);
    /**
     * Performs binary functor on the corresponding cell
     * @param left : [rows x cols]
     * @param right : [rows x cols]
     */
    void EachCellOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad = nullptr);
    /**
     * Performs binary functor on the each row
     * @param left [rows x cols]
     * @param right [1 x cols]
     */
    void RowOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad = nullptr);
    /**
     * Performs binary functor on the each column
     * @param left [rows x cols]
     * @param right [rows x 1]
     */
    void ColOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad = nullptr);
    /**
     * Performs binary functor on the each cell
     * @param left [rows x cols]
     * @param right [1 x 1]
     */
    void CellOperator(const Matrix2D &left, const Matrix2D &right, f32 (*functor)(const f32, const f32), const Matrix2D *grad = nullptr);
    /**
     * Merges all rows into one according to the binary functor
     * @param left [ANY x cols]
     */
    void MergeRowsOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32), const Matrix2D *grad = nullptr, f32(*initFunctor)(const f32) = nullptr);
    /**
     * Merges all columns into one according to the binary functor
     * @param left [rows x ANY]
     */
    void MergeColsOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32), const Matrix2D *grad = nullptr, f32(*initFunctor)(const f32) = nullptr);
    /**
     * Merges cells into one according to the binary functor (increasing columns for each row)
     * @param left [ANY x ANY]
     */
    void MergeCellsOperator(const Matrix2D &left, f32 (*functor)(const f32, const f32), const Matrix2D *grad = nullptr, f32(*initFunctor)(const f32) = nullptr);
    /**
     * Performs matrix multiplication
     * @param left [rows x ANY]
     * @param right [ANY x cols]
     */
    void MultiplyOperator(const Matrix2D &left, const Matrix2D &right);
};

