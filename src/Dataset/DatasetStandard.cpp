#include <fstream>
#include "../../include/Dataset/DatasetStandard.h"
#include <climits>
#include <iostream>

template<class T>
inline T DatasetStandard::ConvertEndian(T value) {
    char *data = reinterpret_cast<char *>(&value);

    T tmp;
    char *tmp_data = reinterpret_cast<char *>(&tmp);

    for (int i = 0; i < sizeof(value); ++i)
        tmp_data[i] = data[sizeof(value) - i - 1];
    return tmp;
}

DatasetStandard::DatasetStandard(const char* filename, size_t batchSize, f32 testCoef, f32 validationCoef) : batchSize(batchSize) {
    size_t trainSamples; // size of train set
    size_t validationSamples; // size of validation set
    size_t testSamples; // size of test set
    std::ifstream file(filename, std::ios::binary);
    // u64 endian check = 4221
    // u32 inputs
    // u32 outputs
    // u64 dataSize
    // f32[inputs] * dataSize
    // f32[outputs] * dataSize

    u64 endian_check;
    u64 dataSize;
    file.read(reinterpret_cast<char*>(&endian_check), sizeof(endian_check));
    file.read(reinterpret_cast<char*>(&inputs), sizeof(inputs));
    file.read(reinterpret_cast<char*>(&outputs), sizeof(outputs));
    file.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    if (endian_check != 4221) {
        if (ConvertEndian(endian_check) != 4221)
            throw std::runtime_error("Invalid file format");
        inputs = ConvertEndian(inputs);
        outputs = ConvertEndian(outputs);
        dataSize = ConvertEndian(dataSize);
    }
    if (testCoef < 1.f)
        testSamples = static_cast<size_t>(static_cast<f64>(dataSize) * static_cast<f64>(testCoef));
    else
        if (static_cast<size_t>(testCoef) < dataSize)
            testSamples = static_cast<size_t>(testCoef);
        else
            throw std::runtime_error("Invalid test coef");

    trainSamples = dataSize - testSamples;
    if (validationCoef < 1.f)
        validationSamples = static_cast<size_t>(static_cast<f64>(trainSamples) * static_cast<f64>(validationCoef));
    else
        if (static_cast<size_t>(validationCoef) < dataSize)
            validationSamples = static_cast<size_t>(validationCoef);
        else
            throw std::runtime_error("Invalid validation coef");

    trainSamples = trainSamples - validationSamples;

    testSamples /= batchSize;
    trainSamples /= batchSize;
    validationSamples /= batchSize;
    size_t train_start = 0, train_end = trainSamples;
    size_t validation_start = trainSamples, validation_end = validation_start + validationSamples;
    size_t test_start = validation_end, test_end = test_start + testSamples;
    if (testSamples <= 0 || trainSamples <= 0)
        throw std::runtime_error("Data size cannot be negative or zero");
    {
        std::vector<f32> data(dataSize * inputs);
        file.read(reinterpret_cast<char *>(data.data()), dataSize * inputs * sizeof(f32));

        for (size_t i = train_start; i < train_end; ++i)
            train_inputs.emplace_back(batchSize, inputs, false).AssignData(data.data() + i * batchSize * inputs);

        for (size_t i = validation_start; i < validation_end; ++i)
            validation_inputs.emplace_back(batchSize, inputs, false).AssignData(data.data() + i * batchSize * inputs);

        for (size_t i = test_start; i < test_end; ++i)
            test_inputs.emplace_back(batchSize, inputs, false).AssignData(data.data() + i * batchSize * inputs);
    }
    {
        std::vector<f32> data(dataSize * static_cast<u64>(outputs));
        file.read(reinterpret_cast<char *>(data.data()), dataSize * static_cast<u64>(outputs) * sizeof(f32));

        for (size_t i = train_start; i < train_end; ++i)
            train_outputs.emplace_back(batchSize, outputs, false).AssignData(data.data() + i * batchSize * outputs);

        for (size_t i = validation_start; i < validation_end; ++i)
            validation_outputs.emplace_back(batchSize, outputs, false).AssignData(data.data() + i * batchSize * outputs);

        for (size_t i = test_start; i < test_end; ++i)
            test_outputs.emplace_back(batchSize, outputs, false).AssignData(data.data() + i * batchSize * outputs);
    }
    std::cout <<
              "Dataset loaded!" << std::endl <<
              "Data size: " << dataSize << std::endl <<
              "Train size: " << trainSamples * batchSize << std::endl <<
              "Validation size: " << validationSamples * batchSize << std::endl <<
              "Test size: " << testSamples * batchSize << std::endl <<
              "Unused size: " << (dataSize - (trainSamples + validationSamples + testSamples) * batchSize) << std::endl;
}

void DatasetStandard::PreprocessMean()
{
    Matrix2D meanMatrix(batchSize, 1);
    for (auto set : { &train_inputs, &validation_inputs, &test_inputs }) {
        for (auto& sample : *set)
        {
            meanMatrix.MergeColsOperator(sample, [](const f32 l, const f32 r) ->f32 {
                return l + r;
                });
            for (size_t i = 0; i < batchSize; ++i)
                meanMatrix.setCell(i, 0, meanMatrix(i, 0) / static_cast<f32>(inputs));
            sample.ColOperator(sample, meanMatrix, [](const f32 l, const f32 r) -> f32 {
                return l - r;
                });
        }
    }
}

std::pair<const Matrix2D &, const Matrix2D &> DatasetStandard::GetTrainSample(bool moveCursor) {
    if (moveCursor)
        ++currentTrainSample;
    if (currentTrainSample >= train_inputs.size())
        currentTrainSample = 0;
    return {train_inputs[currentTrainSample], train_outputs[currentTrainSample]};
}

std::pair<const Matrix2D &, const Matrix2D &> DatasetStandard::GetValidationSample(bool moveCursor) {
    if (moveCursor)
        ++currentValSample;
    if (currentValSample >= validation_inputs.size())
        currentValSample = 0;
    return {validation_inputs[currentValSample], validation_outputs[currentValSample]};
}

std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> DatasetStandard::GetTestSamples() {
    return {test_inputs, test_outputs};
}

std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> DatasetStandard::GetValidationSamples() {
    return {validation_inputs, validation_outputs};
}

size_t DatasetStandard::GetInputs() const {
    return inputs;
}

size_t DatasetStandard::GetOutputs() const {
    return outputs;
}

size_t DatasetStandard::GetBatchSize() const {
    return batchSize;
}

