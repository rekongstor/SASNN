#pragma once

#include <vector>
#include "../stdafx.h"
#include "../Core/Matrix2D.h"
#include "Dataset.h"

class DatasetStandard : public Dataset {
    size_t inputs = 0; // input vector size
    size_t outputs = 0; // output vector size
    size_t batchSize;
    size_t currentTrainSample = 0;
    size_t currentValSample = 0;
    std::vector<Matrix2D> train_inputs; // [batchSize x inputs]
    std::vector<Matrix2D> train_outputs; // [batchSize x outputs]
    std::vector<Matrix2D> validation_inputs; // [batchSize x inputs]
    std::vector<Matrix2D> validation_outputs; // [batchSize x outputs]
    std::vector<Matrix2D> test_inputs; // [batchSize x inputs]
    std::vector<Matrix2D> test_outputs; // [batchSize x outputs]
    template<class T>
    T ConvertEndian(T value);
    std::pair<const Matrix2D &, const Matrix2D &> GetTrainSample(bool moveCursor) override;
    std::pair<const Matrix2D &, const Matrix2D &> GetValidationSample(bool moveCursor) override;
    std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> GetValidationSamples() override;
    std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> GetTestSamples() override;
    [[nodiscard]] size_t GetInputs() const override;
    [[nodiscard]] size_t GetOutputs() const override;
    [[nodiscard]] size_t GetBatchSize() const override;
    void PreprocessMean() override;
public:
    explicit DatasetStandard(const char* filename, size_t batchSize, f32 testCoef, f32 validationCoef);
};

