#include <iostream>
#include "../../include/NeuralNetwork/ClassificationNN.h"
#include "../../include/Layer/Simple/LayerWeights.h"
#include "../../include/Layer/Simple/LayerData.h"
#include "../../include/Layer/Functional/LayerStableSoftMax.h"
#include "../../include/Layer/Functional/LayerCrossEntropyLoss.h"
#include "../../include/Layer/Functional/LayerFullyConnected.h"
#include "../../include/Layer/Functional/LayerL2Reg.h"
#include "../../include/Layer/NLF/LayerLeakyReLU.h"
#include "../../include/Layer/Functional/LayerSum.h"


#define LAYER(Type, ...) Layers.emplace_back(std::dynamic_pointer_cast<Layer>(std::make_shared< Type >( __VA_ARGS__ )))

f32 ClassificationNN::GetAccuracy() {
    return 0;
}

void ClassificationNN::Train(u64 steps) {
    for (u64 i = 0; i < steps; ++i) {
        // Forward propagation
        LossFunction->getGrad()->Fill(.0001f);
        for (auto &Layer : Layers)
            Layer->followProp();
        // Back propagation
        for (auto it = Layers.rbegin(); it != Layers.rend(); ++it)
            (*it)->backProp();

        // Perform gradient descent
        for (auto &Weight : WeightsLayers)
            Weight->subGrad();

        // Clear gradients
        for (auto &Layer : Layers)
            Layer->clearGrad();

        // Load next batch
        auto[Inputs, Outputs] = IO;
        auto[train_inputs, train_outputs] = DataSet.GetTrainSample();
        Inputs->assignData(&train_inputs);
        Outputs->assignData(&train_outputs);
    }
}

ClassificationNN::ClassificationNN(std::vector<u32> &&layers, Dataset &dataset) : DataSet(dataset) {

    auto[train_inputs, train_outputs] = dataset.GetTrainSample();
    auto Input = LAYER(LayerData, train_inputs); // [batch_size x inputs]
    auto Output = LAYER(LayerData, train_outputs); // [batch_size x outputs]
    IO = {Input, Output};
    auto Weights = LAYER(LayerWeights, dataset.GetInputs(), dataset.GetOutputs(), static_cast<f32>(dataset.GetInputs())); // [inputs x outputs]
    WeightsLayers.emplace_back(Weights);
    auto FullyConnected = LAYER(LayerFullyConnected, *Input, *Weights); // [batch_size x outputs]
    auto SoftMax = LAYER(LayerStableSoftMax, *FullyConnected);
    auto CrossEntropyLoss = LAYER(LayerCrossEntropyLoss, *SoftMax, *Output);
    auto L2Regularization = LAYER(LayerL2Reg, *Weights);
    auto Loss = LAYER(LayerSum, *CrossEntropyLoss, *L2Regularization);
    LossFunction = Loss;
}
