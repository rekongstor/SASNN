#pragma once

#include <memory>
#include <map>
#include <list>
#include "NeuralNetwork.h"
#include "../Dataset/Dataset.h"
#include "../Layer/Abstract/Layer.h"

class ClassificationNN : public NeuralNetwork {
    Dataset &DataSet;

    std::vector<std::shared_ptr<Layer>> Layers;
    std::shared_ptr<Layer> LossFunction;
    std::pair<std::shared_ptr<Layer>, std::shared_ptr<Layer>> AccuracyLayer;
    std::pair<std::shared_ptr<Layer>, std::shared_ptr<Layer>> IO;
    std::vector<f32> annealLossValues;
    void ForwardPropagation(Layer *stop_layer = nullptr);
    void Deserialize(const char *filename) override;
    void BackPropagation();
    void GradientDescent();
    void ClearGradients();
    std::map<char, std::shared_ptr<Matrix2D>> HyperParams;
    std::vector<std::shared_ptr<Layer>> WeightLayers;

    explicit ClassificationNN(std::vector<s32> &&layers, Dataset &dataset);
    f32 GetAccuracy(std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> samples);
public:
    void Use(Matrix2D &in, Matrix2D &out) override;
private:
    f32 GetAccuracy(std::pair<const Matrix2D &, const Matrix2D &> sample);
    void ModifyParam(char param_name, f32 value) override;
    void AdaptLearningRate();
    void Serialize(const char *filename) override;
    std::pair<f32, f32> Test() override;
    std::pair<f32, f32> Train() override;
public:
    template<class... Args>
    explicit ClassificationNN(Dataset &dataset, Args &&... args): ClassificationNN({std::forward<Args>(args)...}, dataset) {
    }

};



