#pragma once

#include <vector>
#include "../stdafx.h"
#include "../Core/Matrix2D.h"
#include "Dataset.h"

class DatasetStandard : public Dataset {
    u32 inputs = 0; // input vector size
    u32 outputs = 0; // output vector size
    size_t trainSamples; // size of train set
    size_t testSamples; // size of test set
    size_t batchSize;
    size_t currentTrainSample = -1;
    size_t currentTestSample = -1;
    std::vector<Matrix2D> train_inputs; // [batchSize x inputs]
    std::vector<Matrix2D> train_outputs; // [batchSize x outputs]
    std::vector<Matrix2D> test_inputs; // [batchSize x inputs]
    std::vector<Matrix2D> test_outputs; // [batchSize x outputs]
    template<class T>
    T ConvertEndian(T value);
    std::pair<const Matrix2D &, const Matrix2D &> GetTrainSample() override;
    std::pair<const Matrix2D &, const Matrix2D &> GetTestSample() override;
    [[nodiscard]] u32 GetInputs() const override;
    [[nodiscard]]  u32 GetOutputs() const override;
public:
    explicit DatasetStandard(std::ifstream &&file, size_t batchSize = 512, f32 testCoef = 0.1f);
};



