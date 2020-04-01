#include <iostream>
#include <fstream>
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
#include "../../include/Layer/Simple/LayerWeightsDecorators/Initializer/InitializerXavier.h"
#include "../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentStochastic.h"
#include "../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentMomentum.h"
#include "../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentAdaGrad.h"
#include "../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentAdam.h"


#define ADD_LAYER(Type, ...) Layers.emplace_back( Type )
#define ADD_PARAM(Param, Value) HyperParams.emplace( Param , std::make_shared<Matrix2D>( Value ))


f32 ClassificationNN::Test() {
    return GetAccuracy(DataSet.GetTestSamples());
}

f32 ClassificationNN::Train(u64 steps) {
    f32 acc = GetAccuracy(DataSet.GetValidationSamples());;
    for (u64 i = 0; i < steps; ++i) {
        // Load next batch
        auto[Inputs, Outputs] = IO;
        auto[train_inputs, train_outputs] = DataSet.GetTrainSample(true);
        Inputs->assignData(&train_inputs);
        Outputs->assignData(&train_outputs);

        ForwardPropagation();
        BackPropagation();
        GradientDescent();
        ClearGradients();
    }
    return acc;
}

ClassificationNN::ClassificationNN(std::vector<u32> &&layers, Dataset &dataset) : DataSet(dataset) {
    // Setting hyper-parameters. Modifiable
    ADD_PARAM('l', 1.f);
    ADD_PARAM('g', 1.f);

    // Setting dataset layers
    auto[train_inputs, train_outputs] = DataSet.GetTrainSample(false);
    auto Input = ADD_LAYER(new LayerData(train_inputs)); // [batch_size x inputs]
    auto Output = ADD_LAYER(new LayerData(train_outputs)); // [batch_size x outputs]
    IO = {Input, Output};

    // Setting hyper-parameter layers
    auto L2RegParam = ADD_LAYER(new LayerData(*HyperParams['l']));
    // Setting FullyConnected architecture
//    auto Weights = ADD_LAYER(LayerWeights, dataset.GetInputs(), dataset.GetOutputs(),
//                             SP_CAST(DecoratorInitializer, InitializerXavier, static_cast<f32>(dataset.GetInputs())),
//                             SP_CAST(DecoratorGradientDescent, GradientDescentStochastic));
//
//    Layers.emplace_back(std::dynamic_pointer_cast<Layer>(std::make_shared<LayerWeights>(
//            dataset.GetInputs(), dataset.GetOutputs(),
//            std::dynamic_pointer_cast<DecoratorInitializer>(std::make_shared<InitializerXavier>(static_cast<f32>(dataset.GetInputs()))),
//            std::dynamic_pointer_cast<DecoratorGradientDescent>(std::make_shared<GradientDescentStochastic>())
//    )));

    auto Weights = ADD_LAYER(new LayerWeights(dataset.GetInputs(), dataset.GetOutputs(), new InitializerXavier(static_cast<f32>(dataset.GetInputs())), new GradientDescentAdaGrad));

    WeightsLayers.emplace_back(Weights);
    auto FullyConnected = ADD_LAYER(new LayerFullyConnected(*Input, *Weights)); // [batch_size x outputs]

    // Setting Loss function
    auto SoftMax = ADD_LAYER(new LayerStableSoftMax(*FullyConnected, true));
    auto CrossEntropyLoss = ADD_LAYER(new LayerCrossEntropyLoss(*SoftMax, *Output));
    auto L2Regularization = ADD_LAYER(new LayerL2Reg(*Weights, *L2RegParam));
    auto Loss = ADD_LAYER(new LayerSum(*CrossEntropyLoss, *L2Regularization));
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
        ForwardPropagation(AccuracyLayer.second.get());
        AccuracyLayer.first->followProp();
        accuracy += AccuracyLayer.first->getData()(0, 0);
    }
    accuracy /= static_cast<f32>(DataSet.GetBatchSize() * inputs.size());
    return accuracy;
}

void ClassificationNN::ForwardPropagation(Layer *stop_layer) {
    // Forward propagation
    for (auto &Layer : Layers) {
        Layer->followProp();
        if (Layer.get() == stop_layer)
            break;
    }
}

void ClassificationNN::BackPropagation() {
    // Back propagation
    LossFunction->getGrad()->Fill(1.f);
    for (auto it = Layers.rbegin(); it != Layers.rend(); ++it)
        (*it)->backProp();
}

void ClassificationNN::ClearGradients() {
    // Clear gradients
    for (auto &Layer : Layers)
        Layer->clearGrad();
}

void ClassificationNN::GradientDescent() {
    // Perform gradient descent
    for (auto &Weight : WeightsLayers)
        Weight->subGrad((*HyperParams['g'])(0, 0));
}

void ClassificationNN::Serialize(const char *filename) {
    std::ofstream out(filename);
    for (auto &W : WeightsLayers)
        for (size_t j = 0; j < W->getData().getCols(); ++j)
            for (size_t i = 0; i < W->getData().getRows(); ++i)
                out << W->getData()(i, j) << std::endl;
}
