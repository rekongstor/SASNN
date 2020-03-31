#include <fstream>
#include "../../include/Dataset/DatasetStandard.h"
#include <climits>

template<class T>
inline T DatasetStandard::ConvertEndian(T value) {
    char *data = reinterpret_cast<char *>(&value);

    T tmp;
    char *tmp_data = reinterpret_cast<char *>(&tmp);

    for (int i = 0; i < sizeof(value); ++i)
        tmp_data[i] = data[sizeof(value) - i - 1];
    return tmp;
}

DatasetStandard::DatasetStandard(std::ifstream &&file, size_t batchSize, f32 testCoef) : batchSize(batchSize) {
    // u64 endian check = 4221
    // u32 inputs
    // u32 outputs
    // u64 dataSize
    // f32[inputs] * dataSize
    // f32[outputs] * dataSize
    u64 endian_check;
    u64 dataSize;
    file.read(reinterpret_cast<char *>(&endian_check), sizeof(endian_check));
    file.read(reinterpret_cast<char *>(&inputs), sizeof(inputs));
    file.read(reinterpret_cast<char *>(&outputs), sizeof(outputs));
    file.read(reinterpret_cast<char *>(&dataSize), sizeof(dataSize));
    if (endian_check != 4221) {
        if (ConvertEndian(endian_check) != 4221)
            throw std::runtime_error("Invalid file format");
        inputs = ConvertEndian(inputs);
        outputs = ConvertEndian(outputs);
        dataSize = ConvertEndian(dataSize);
    }

    testSamples = dataSize * testCoef;
    trainSamples = dataSize - testSamples;
    testSamples /= batchSize;
    trainSamples /= batchSize;
    if (testSamples <= 0 || trainSamples <= 0)
        throw std::runtime_error("Data size cannot be negative or zero");
    {
        std::vector<f32> data(dataSize * inputs);
        file.read(reinterpret_cast<char *>(data.data()), dataSize * inputs * sizeof(f32));
        for (size_t i = 0; i < trainSamples; ++i)
            train_inputs.emplace_back(std::move(Matrix2D(batchSize, inputs, false))).AssignData(data.data() + i * batchSize * inputs);
        for (size_t i = trainSamples; i < trainSamples + testSamples; ++i)
            test_inputs.emplace_back(std::move(Matrix2D(batchSize, inputs, false))).AssignData(data.data() + i * batchSize * inputs);
    }
    {
        std::vector<f32> data(dataSize * outputs);
        file.read(reinterpret_cast<char *>(data.data()), dataSize * outputs * sizeof(f32));
        for (size_t i = 0; i < trainSamples; ++i)
            train_outputs.emplace_back(std::move(Matrix2D(batchSize, outputs, false))).AssignData(data.data() + i * batchSize * outputs);
        for (size_t i = trainSamples; i < trainSamples + testSamples; ++i)
            test_outputs.emplace_back(std::move(Matrix2D(batchSize, outputs, false))).AssignData(data.data() + i * batchSize * outputs);
    }
}

std::pair<const Matrix2D &, const Matrix2D &> DatasetStandard::GetTrainSample() {
    ++currentTrainSample;
    if (currentTrainSample >= train_inputs.size())
        currentTrainSample = -1;
    return {train_inputs[currentTrainSample], train_outputs[currentTrainSample]};
}

std::pair<const Matrix2D &, const Matrix2D &> DatasetStandard::GetTestSample() {
    ++currentTestSample;
    if (currentTestSample >= test_inputs.size())
        currentTestSample = -1;
    return {test_inputs[currentTestSample], test_outputs[currentTestSample]};

}

u32 DatasetStandard::GetInputs() const {
    return inputs;
}

u32 DatasetStandard::GetOutputs() const {
    return outputs;
}
