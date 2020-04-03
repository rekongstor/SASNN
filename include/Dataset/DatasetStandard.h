#pragma once

#include <vector>
#include "../stdafx.h"
#include "../Core/Matrix2D.h"
#include "Dataset.h"

class DatasetStandard : public Dataset {
    u32 inputs = 0; // input vector size
    u32 outputs = 0; // output vector size
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
    std::pair< Matrix2D &,  Matrix2D &> GetTrainSample(bool moveCursor) override;
    std::pair< Matrix2D &,  Matrix2D &> GetValidationSample(bool moveCursor) override;
    std::pair< std::vector<Matrix2D> &,  std::vector<Matrix2D> &> GetValidationSamples() override;
    std::pair< std::vector<Matrix2D> &,  std::vector<Matrix2D> &> GetTestSamples() override;
    [[nodiscard]] u32 GetInputs() const override;
    [[nodiscard]] u32 GetOutputs() const override;
    [[nodiscard]] u32 GetBatchSize() const override;
    void PreprocessMean() override;
public:
    explicit DatasetStandard(const char* filename, size_t batchSize, f32 testCoef, f32 validationCoef);
};

