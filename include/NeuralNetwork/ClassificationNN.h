#pragma once

#include <memory>
#include <map>
#include "NeuralNetwork.h"
#include "../Dataset/Dataset.h"
#include "../Layer/Abstract/Layer.h"

class ClassificationNN : public NeuralNetwork {
    Dataset &DataSet;

    std::vector<std::shared_ptr<Layer>> Layers;
    std::vector<std::shared_ptr<Layer>> WeightsLayers;
    std::shared_ptr<Layer> LossFunction;
    std::pair<std::shared_ptr<Layer>, std::shared_ptr<Layer>> AccuracyLayer;
    std::pair<std::shared_ptr<Layer>, std::shared_ptr<Layer>> IO;
    void ForwardPropagation(Layer *stop_layer = nullptr);
    void BackPropagation();
    std::map<char, std::shared_ptr<Matrix2D>> HyperParams;

    explicit ClassificationNN(std::vector<u32> &&layers, Dataset &dataset);
    f32 GetAccuracy(std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> samples);
    void ModifyParam(char param_name, f32 value) override;
    f32 Test() override;
    f32 Train(u64 steps) override;
public:
    template<class... Args>
    explicit ClassificationNN(Dataset &dataset, Args &&... args): ClassificationNN({std::forward<Args>(args)...}, dataset) {
    }

    void Serialize(const char *filename) override;
};



