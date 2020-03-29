#pragma once

#include <vector>
#include "../stdafx.h"
#include "../Core/Matrix2D.h"

class Dataset {
    u32 inputs; // input vector size
    u32 outputs; // output vector size
    size_t trainSize; // size of train set
    size_t testSize; // size of test set
    size_t batchSize;
    std::vector<Matrix2D> train_inputs; // [batchSize x inputs]
    std::vector<Matrix2D> train_outputs; // [batchSize x outputs]
    std::vector<Matrix2D> test_inputs; // [batchSize x inputs]
    std::vector<Matrix2D> test_outputs; // [batchSize x outputs]
    void CopyData(std::vector<Matrix2D> &dst, const f32 *src, u32 size, u32 samples);
public:
    Dataset(u32 inputs, u32 outputs, const f32 *trainInputs_data, const f32 *trainOutputs_data, const f32 *testInputs_data, const f32 *testOutputs_data, size_t batchSize,
            size_t trainBatches,
            size_t testBatches);
    [[nodiscard]] std::pair<const Matrix2D &, const Matrix2D &> GetTrainSample() const;
    [[nodiscard]] std::pair<const Matrix2D &, const Matrix2D &> GetTestSample() const;
};



