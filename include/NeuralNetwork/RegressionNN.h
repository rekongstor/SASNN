#pragma once

#include <memory>
#include <map>
#include <list>
#include "NeuralNetwork.h"
#include "../Dataset/Dataset.h"
#include "../Layer/Abstract/Layer.h"

class RegressionNN : public NeuralNetwork {
    Dataset &DataSet;

    std::vector<std::shared_ptr<Layer>> Layers;
    std::shared_ptr<Layer> LossFunction;
    std::pair<std::shared_ptr<Layer>, std::shared_ptr<Layer>> AccuracyLayer;
    std::pair<std::shared_ptr<Layer>, std::shared_ptr<Layer>> IO;
    std::vector<f32> annealLossValues;
    void ForwardPropagation(Layer *stop_layer = nullptr);
    void BackPropagation();
    void GradientDescent();
    void ClearGradients();
    std::map<char, std::shared_ptr<Matrix2D>> HyperParams;
    std::vector<std::shared_ptr<Layer>> WeightLayers;
    void Deserialize(const char *filename) override;
    void Serialize(const char *filename) override;
    void Use() override;

    explicit RegressionNN(std::vector<s32> &&layers, Dataset &dataset);
    f32 GetAccuracy(std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> samples);
    f32 GetAccuracy(std::pair<const Matrix2D &, const Matrix2D &> sample);
    void ModifyParam(char param_name, f32 value) override;
    std::pair<f32, f32> Test() override;
    std::pair<f32, f32> Train() override;
public:
    template<class... Args>
    explicit RegressionNN(Dataset &dataset, Args &&... args): RegressionNN({std::forward<Args>(args)...}, dataset) {
    }

    void AdaptLearningRate();
};



