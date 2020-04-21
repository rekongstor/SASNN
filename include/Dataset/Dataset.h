#pragma once


#include "../Core/Matrix2D.h"

class Dataset {
public:
    [[nodiscard]] virtual size_t GetInputs() const = 0;
    [[nodiscard]] virtual size_t GetOutputs() const = 0;
    [[nodiscard]] virtual size_t GetBatchSize() const = 0;
    [[nodiscard]] virtual std::pair<const Matrix2D &, const Matrix2D &> GetTrainSample(bool moveCursor) = 0;
    [[nodiscard]] virtual std::pair<const Matrix2D &, const Matrix2D &> GetValidationSample(bool moveCursor) = 0;
    [[nodiscard]] virtual std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> GetValidationSamples() = 0;
    [[nodiscard]] virtual std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> GetTestSamples() = 0;
    [[nodiscard]] virtual std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> GetTrainSamples() = 0;
    virtual void PreprocessMean() = 0;
    virtual ~Dataset() = default;
};



