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
        LossFunction->getGrad()->Fill(HyperParams["GradStep"](0, 0));
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
    // Setting hyper-parameters. Modifiable
    HyperParams.insert({"L2Reg", Matrix2D(0.01f)});
    HyperParams.insert({"GradStep", Matrix2D(0.0001f)});

    // Setting dataset layers
    auto[train_inputs, train_outputs] = dataset.GetTrainSample();
    auto Input = LAYER(LayerData, train_inputs); // [batch_size x inputs]
    auto Output = LAYER(LayerData, train_outputs); // [batch_size x outputs]
    IO = {Input, Output};

    // Setting hyper-parameter layers
    auto L2RegParam = LAYER(LayerData, HyperParams["L2Reg"]);

    // Setting FullyConnected architecture
    auto Weights = LAYER(LayerWeights, dataset.GetInputs(), dataset.GetOutputs(), static_cast<f32>(dataset.GetInputs())); // [inputs x outputs]
    WeightsLayers.emplace_back(Weights);
    auto FullyConnected = LAYER(LayerFullyConnected, *Input, *Weights); // [batch_size x outputs]

    // Setting Loss function
    auto SoftMax = LAYER(LayerStableSoftMax, *FullyConnected);
    auto CrossEntropyLoss = LAYER(LayerCrossEntropyLoss, *SoftMax, *Output);
    auto L2Regularization = LAYER(LayerL2Reg, *Weights, *L2RegParam);
    auto Loss = LAYER(LayerSum, *CrossEntropyLoss, *L2Regularization);
    LossFunction = Loss;
}

void ClassificationNN::ModifyParam(const char *param_name, f32 value) {
    auto m = HyperParams.find(param_name);
    if (m != HyperParams.end())
        m->second.setCell(0, 0, value);
    else
        std::cout << "Invalid parameter name: " << param_name << std::endl;
}
