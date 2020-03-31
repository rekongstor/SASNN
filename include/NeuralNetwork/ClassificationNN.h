#pragma once

#include <memory>
#include "NeuralNetwork.h"
#include "../Dataset/Dataset.h"
#include "../Layer/Abstract/Layer.h"

class ClassificationNN : public NeuralNetwork {
    std::vector<std::shared_ptr<Layer>> Layers;
    std::vector<std::shared_ptr<Layer>> WeightsLayers;
    std::shared_ptr<Layer> LossFunction;
    std::pair<std::shared_ptr<Layer>, std::shared_ptr<Layer>> IO;
    Dataset& DataSet;
    explicit ClassificationNN(std::vector<u32> &&layers, Dataset &dataset);
public:
    f32 GetAccuracy() override;
    void Train(u64 steps) override;

    template<class... Args>
    explicit ClassificationNN(Dataset &dataset, Args &&... args): ClassificationNN({std::forward<Args>(args)...}, dataset) {
    }
};



