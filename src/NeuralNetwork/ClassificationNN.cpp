#include <iostream>
#include "../../include/NeuralNetwork/ClassificationNN.h"
#include "../../include/Layer/Simple/LayerWeights.h"
#include "../../include/Layer/Simple/LayerData.h"
#include "../../include/Layer/Functional/LayerStableSoftMax.h"
#include "../../include/Layer/Functional/LayerCrossEntropyLoss.h"
#include "../../include/Layer/Functional/LayerFullyConnected.h"
#include "../../include/Layer/Functional/LayerL2Reg.h"
#include "../../include/Layer/Accuracy/LayerClassificationAccuracy.h"
#include "../../include/Layer/NLF/LayerLeakyReLU.h"
#include "../../include/Layer/Functional/LayerSum.h"


#define LAYER(Type, ...) Layers.emplace_back(std::dynamic_pointer_cast<Layer>(std::make_shared< Type >( __VA_ARGS__ )))
#define PARAM(Param, Value) HyperParams.emplace( Param , std::make_shared<Matrix2D>( Value ))


f32 ClassificationNN::Test() {
    return GetAccuracy(DataSet.GetTestSamples());
}

f32 ClassificationNN::Train(u64 steps) {
    for (u64 i = 0; i < steps; ++i) {
        // Load next batch
        auto[Inputs, Outputs] = IO;
        auto[train_inputs, train_outputs] = DataSet.GetTrainSample(true);
        Inputs->assignData(&train_inputs);
        Outputs->assignData(&train_outputs);

        FollowPropagation();
        BackPropagation();
    }
    return GetAccuracy(DataSet.GetValidationSamples());
}

ClassificationNN::ClassificationNN(std::vector<u32> &&layers, Dataset &dataset) : DataSet(dataset) {
    // Setting hyper-parameters. Modifiable
    PARAM('l', 1.f);
    PARAM('g', 0.01f);

    // Setting dataset layers
    auto[train_inputs, train_outputs] = DataSet.GetTrainSample(false);
    auto Input = LAYER(LayerData,train_inputs); // [batch_size x inputs]
    auto Output = LAYER(LayerData, train_outputs); // [batch_size x outputs]
    IO = {Input, Output};

    // Setting hyper-parameter layers
    auto L2RegParam = LAYER(LayerData, *HyperParams['l']);

    // Setting FullyConnected architecture
    auto Weights = LAYER(LayerWeights, dataset.GetInputs(), dataset.GetOutputs(), static_cast<f32>(dataset.GetInputs())); // [inputs x outputs]
    WeightsLayers.emplace_back(Weights);
    auto FullyConnected = LAYER(LayerFullyConnected, *Input, *Weights); // [batch_size x outputs]

    // Setting Loss function
    auto SoftMax = LAYER(LayerStableSoftMax, *FullyConnected, true);
    auto CrossEntropyLoss = LAYER(LayerCrossEntropyLoss, *SoftMax, *Output);
    auto L2Regularization = LAYER(LayerL2Reg, *Weights, *L2RegParam);
    auto Loss = LAYER(LayerSum, *CrossEntropyLoss, *L2Regularization);
    LossFunction = Loss;

    // Setting Accuracy layer
    AccuracyLayer = {std::dynamic_pointer_cast<Layer>(std::make_shared<LayerClassificationAccuracy>(*SoftMax, *Output, true)), SoftMax};
}

void ClassificationNN::ModifyParam(char param_name, f32 value) {
    auto m = HyperParams.find(param_name);
    if (m != HyperParams.end())
        (*m->second).setCell(0, 0, value);
    else
        std::cout << "Invalid parameter name: " << param_name << std::endl;
}

f32 ClassificationNN::GetAccuracy(std::pair<const std::vector<Matrix2D> &, const std::vector<Matrix2D> &> samples) {
    auto[inputs, outputs] = samples;
    f32 accuracy = 0.f;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto[Inputs, Outputs] = IO;
        Inputs->assignData(&inputs[i]);
        Outputs->assignData(&outputs[i]);
        FollowPropagation(AccuracyLayer.second.get());
        AccuracyLayer.first->followProp();
        accuracy += AccuracyLayer.first->getData()(0,0);
    }
    accuracy /= static_cast<f32>(DataSet.GetBatchSize() * inputs.size());
    return accuracy;
}

void ClassificationNN::FollowPropagation(Layer *stop_layer) {
    // Forward propagation
    for (auto &Layer : Layers) {
        Layer->followProp();
        if (Layer.get() == stop_layer)
            break;
    }
}

void ClassificationNN::BackPropagation() {
    // Back propagation
    LossFunction->getGrad()->Fill((*HyperParams['g'])(0, 0));
    for (auto it = Layers.rbegin(); it != Layers.rend(); ++it)
        (*it)->backProp();

    // Perform gradient descent
    for (auto &Weight : WeightsLayers)
        Weight->subGrad();

    // Clear gradients
    for (auto &Layer : Layers)
        Layer->clearGrad();
}
