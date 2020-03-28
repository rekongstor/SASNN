#pragma once

#include <vector>
#include "stdafx.h"

class Matrix2D {
    size_t rows;
    size_t cols;
    std::vector<f32> data;
public:
    Matrix2D(size_t rows, size_t cols);
    f32 &operator()(size_t row, size_t col);

    [[nodiscard]] size_t getRows() const;
    [[nodiscard]] size_t getCols() const;
    void copyRow(size_t row, const f32 *data);
};

