#include "../include/Dataset.h"

Dataset::Dataset(u32 inputs, u32 outputs, const f32 *trainInputs_data, const f32 *trainOutputs_data, const f32 *testInputs_data, const f32 *testOutputs_data, size_t batchSize,
                 size_t trainBatches, size_t testBatches)
        : inputs(inputs),
          outputs(outputs),
          batchSize(batchSize),
          trainSize(trainBatches * batchSize),
          testSize(testBatches * batchSize) {
    CopyData(train_inputs, trainInputs_data, inputs, trainBatches);
    CopyData(train_outputs, trainOutputs_data, outputs, trainBatches);
    CopyData(test_inputs, testInputs_data, inputs, testBatches);
    CopyData(test_outputs, testOutputs_data, outputs, testBatches);
}

void Dataset::CopyData(std::vector<Matrix2D> &dst, const f32 *src, u32 size, u32 samples) {
    for (u32 s = 0; s < samples; ++s) {
        auto &Sample = dst.emplace_back(std::move(Matrix2D(batchSize, size, false)));
        for (size_t i = 0; i < batchSize; ++i)
            Sample.copyRow(i, src + i * size + s * batchSize * size);
    }
}

std::pair<const Matrix2D &, const Matrix2D &> Dataset::GetTrainSample() const {
    static size_t cur_sample = 0;
    if (cur_sample >= train_inputs.size())
        cur_sample = 0;
    return {train_inputs[cur_sample], train_outputs[cur_sample]};
}

std::pair<const Matrix2D &, const Matrix2D &> Dataset::GetTestSample() const {
    static size_t cur_sample = 0;
    if (cur_sample >= test_inputs.size())
        cur_sample = 0;
    return {test_inputs[cur_sample], test_outputs[cur_sample]};
}

