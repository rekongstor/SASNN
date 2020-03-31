#pragma once

#include <memory>
#include <map>
#include "NeuralNetwork.h"
#include "../Dataset/Dataset.h"
#include "../Layer/Abstract/Layer.h"

class ClassificationNN : public NeuralNetwork {
    Dataset& DataSet;

    std::vector<std::shared_ptr<Layer>> Layers;
    std::vector<std::shared_ptr<Layer>> WeightsLayers;
    std::shared_ptr<Layer> LossFunction;
    std::pair<std::shared_ptr<Layer>, std::shared_ptr<Layer>> IO;

    std::map<std::string,Matrix2D> HyperParams;

    explicit ClassificationNN(std::vector<u32> &&layers, Dataset &dataset);
    void ModifyParam(const char *param_name, f32 value) override;
    f32 GetAccuracy() override;
    void Train(u64 steps) override;
public:
    template<class... Args>
    explicit ClassificationNN(Dataset &dataset, Args &&... args): ClassificationNN({std::forward<Args>(args)...}, dataset) {
    }
};



