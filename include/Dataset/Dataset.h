#pragma once


#include "../Core/Matrix2D.h"

class Dataset {
public:
    [[nodiscard]] virtual u32 GetInputs() const = 0;
    [[nodiscard]] virtual u32 GetOutputs() const = 0;
    [[nodiscard]] virtual std::pair<const Matrix2D &, const Matrix2D &> GetTrainSample() = 0;
    [[nodiscard]] virtual std::pair<const Matrix2D &, const Matrix2D &> GetTestSample() = 0;
    virtual ~Dataset() = default;
};



