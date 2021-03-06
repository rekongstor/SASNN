#include <iostream>
#include <fstream>
#include "../../include/NeuralNetwork/ClassificationNN.h"
#include "../../include/Layer/Simple/LayerWeights.h"
#include "../../include/Layer/Simple/LayerData.h"
#include "../../include/Layer/Resulting/LayerStableSoftMax.h"
#include "../../include/Layer/Loss/LayerCrossEntropyLoss.h"
#include "../../include/Layer/Functional/LayerFullyConnected.h"
#include "../../include/Layer/Functional/LayerL2Reg.h"
#include "../../include/Layer/Accuracy/LayerClassificationAccuracy.h"
#include "../../include/Layer/NLF/LayerLeakyReLU.h"
#include "../../include/Layer/NLF/LayerReLU.h"
#include "../../include/Layer/NLF/LayerTanh.h"
#include "../../include/Layer/NLF/LayerSigmoid.h"
#include "../../include/Layer/Functional/LayerSum.h"
#include "../../include/Layer/Functional/LayerBatchNormalization.h"
#include "../../include/Layer/LayerDecorators/Initializer/InitializerXavier.h"
#include "../../include/Layer/LayerDecorators/Initializer/InitializerZero.h"
#include "../../include/Layer/LayerDecorators/Initializer/InitializerUniform.h"
#include "../../include/Layer/LayerDecorators/GradientDescent/GradientDescentStochastic.h"
#include "../../include/Layer/LayerDecorators/GradientDescent/GradientDescentMomentum.h"
#include "../../include/Layer/LayerDecorators/GradientDescent/GradientDescentAdaGrad.h"
#include "../../include/Layer/LayerDecorators/GradientDescent/GradientDescentAdam.h"
#include "../../include/Layer/LayerDecorators/GradientDescent/GradientDescentRMSProp.h"


#define ADD_LAYER(Type, ...) Layers.emplace_back( Type )
#define ADD_PARAM(Param, Value) HyperParams.emplace( Param , std::make_shared<Matrix2D>( Value ))
#define GET_PARAM(Param) (*HyperParams[ Param ])(0, 0)


std::pair<f32, f32> ClassificationNN::Test() {
    return {GetAccuracy(DataSet.GetValidationSamples()), GetAccuracy(DataSet.GetTestSamples())};
}

std::pair<f32, f32> ClassificationNN::Train() {
    for (u64 i = 0; i < std::round(GET_PARAM('s')); ++i) {
        // Load next batch
        auto[Inputs, Outputs] = IO;
        auto[train_inputs, train_outputs] = DataSet.GetTrainSample(true);
        Inputs->assignData(&train_inputs);
        Outputs->assignData(&train_outputs);
        ForwardPropagation();
        BackPropagation();
        GradientDescent();
        ClearGradients();

        annealLossValues[i] = LossFunction->getData()(0, 0);
    }
    AdaptLearningRate();

    f32 trainAcc = GetAccuracy(DataSet.GetTrainSample(false));
    f32 valAcc = GetAccuracy(DataSet.GetValidationSample(true));


    return {trainAcc, valAcc};
}

void ClassificationNN::AdaptLearningRate() {
    f32 annMean = 0.f;
    for (auto &&lv : annealLossValues)
        annMean += lv;
    annMean /= GET_PARAM('s');
    f32 x_avr = GET_PARAM('s') / 2.f;
    f32 y_avr = annMean;
    f32 top = 0.f, btm = 0.f;
    f32 a;
    for (size_t i = 0; i < annealLossValues.size(); ++i) {
        top += (static_cast<f32>(i) - x_avr) * (annealLossValues[i] - y_avr);
        btm += (static_cast<f32>(i) - x_avr) * (static_cast<f32>(i) - x_avr);
    }
    a = top / btm;

    if (annealLossValues.front() < annMean || a > 0.0f) {
        ModifyParam('l', GET_PARAM('l') * GET_PARAM('a'));
        printf("Loss is growing! Learning rate was decreased! Current value: %f\n", GET_PARAM('l'));
    } else if (a < -0.1f) {
        ModifyParam('l', GET_PARAM('l') * (2.f - GET_PARAM('a')));
        printf("Loss is decreasing! Learning rate was increased! Current value: %f\n", GET_PARAM('l'));
    }
}

ClassificationNN::ClassificationNN(std::vector<s32> &&layers, Dataset &dataset) : DataSet(dataset) {
    // Setting hyper-parameters. Modifiable
    ADD_PARAM('l', 0.0003f); // Learning rate
    ADD_PARAM('r', 1.f); // Regularization
    ADD_PARAM('s', 50.f); // Steps
    ADD_PARAM('a', 0.5f); // Annealing multiplier
    annealLossValues.resize(static_cast<size_t>(std::round(GET_PARAM('s'))));

    // Setting dataset layers
    auto[train_inputs, train_outputs] = DataSet.GetTrainSample(false);
    auto Input = ADD_LAYER(new LayerData(train_inputs)); // [batch_size x inputs]
    auto Output = ADD_LAYER(new LayerData(train_outputs)); // [batch_size x outputs]
    IO = {Input, Output};
    // Setting hyper-parameter layers
    auto L2RegParam = ADD_LAYER(new LayerData(*HyperParams['r']));

    std::vector<std::shared_ptr<Layer>> RegularizationLayers;
    // Initializing NN layers
    std::shared_ptr<Layer> InputNeurons; // [batch_size x inputs]
    InputNeurons = Input;
    for (auto outputs : layers) {
        auto Weights = ADD_LAYER(
                new LayerWeights(InputNeurons->getData().getCols(), static_cast<size_t>(outputs),
                                 new InitializerXavier(static_cast<f32>(InputNeurons->getData().getCols())),
                                 new GradientDescentAdam)); // [inputs x outputs]
        auto FullyConnected = ADD_LAYER(new LayerFullyConnected(*InputNeurons, *Weights)); // [batch_size x outputs]
        auto Biases = ADD_LAYER(
                new LayerWeights(1, static_cast<size_t>(outputs),
                                 new InitializerXavier(static_cast<f32>(InputNeurons->getData().getCols())),
                                 new GradientDescentAdam)); // [1 x outputs]
        WeightLayers.push_back(Weights);
        WeightLayers.push_back(Biases);
        auto Neurons = ADD_LAYER(new LayerSum(*FullyConnected, *Biases));
        // Batch Normalization
        auto BN_Beta = ADD_LAYER(
                new LayerWeights(1, 1,
                                 new InitializerZero(),
                                 new GradientDescentAdam)); // [1 x outputs]
        auto BN_Gamma = ADD_LAYER(
                new LayerWeights(1, 1,
                                 new InitializerUniform(1.f,1.f),
                                 new GradientDescentAdam)); // [1 x outputs]
        WeightLayers.push_back(BN_Beta);
        WeightLayers.push_back(BN_Gamma);
        auto BatchNormalization = ADD_LAYER(new LayerBatchNormalization(*Neurons, *BN_Beta, *BN_Gamma));
        // ReLU
        auto ReLU = ADD_LAYER(new LayerLeakyReLU(*BatchNormalization));
        auto L2Regularization = ADD_LAYER(new LayerL2Reg(*Weights, *L2RegParam));
        RegularizationLayers.push_back(L2Regularization);
        auto L2RegularizationBias = ADD_LAYER(new LayerL2Reg(*Biases, *L2RegParam));
        RegularizationLayers.push_back(L2RegularizationBias);
        InputNeurons = ReLU;
    }

    // Classification Layer
    auto Weights = ADD_LAYER(
            new LayerWeights(InputNeurons->getData().getCols(), dataset.GetOutputs(),
                             new InitializerXavier(static_cast<f32>(dataset.GetInputs())),
                             new GradientDescentAdam));
    auto FullyConnected = ADD_LAYER(new LayerFullyConnected(*InputNeurons, *Weights)); // [batch_size x outputs]

    auto Biases = ADD_LAYER(
            new LayerWeights(1, FullyConnected->getData().getCols(),
                             new InitializerXavier(static_cast<f32>(InputNeurons->getData().getCols())),
                             new GradientDescentAdam)); // [1 x outputs]
    WeightLayers.push_back(Weights);
    WeightLayers.push_back(Biases);
    auto Neurons = ADD_LAYER(new LayerSum(*FullyConnected, *Biases));

    auto L2Regularization = ADD_LAYER(new LayerL2Reg(*Weights, *L2RegParam));
    RegularizationLayers.push_back(L2Regularization);
    auto L2RegularizationBias = ADD_LAYER(new LayerL2Reg(*Biases, *L2RegParam));
    RegularizationLayers.push_back(L2RegularizationBias);

    // Setting Loss function
    auto SoftMax = ADD_LAYER(new LayerStableSoftMax(*Neurons, true));
    auto CrossEntropyLoss = ADD_LAYER(new LayerCrossEntropyLoss(*SoftMax, *Output));
    std::shared_ptr<Layer> Loss = CrossEntropyLoss;
    for (auto &&RL : RegularizationLayers)
        Loss = ADD_LAYER(new LayerSum(*Loss, *RL.get()));

    LossFunction = Loss;

    // Setting Accuracy layer
    AccuracyLayer = {std::dynamic_pointer_cast<Layer>(std::make_shared<LayerClassificationAccuracy>(*SoftMax, *Output, true)), SoftMax};
}

void ClassificationNN::ModifyParam(char param_name, f32 value) {
    switch (param_name) {
        case 's':
            if (std::round(value) < 10) {
                std::cout << "Parameter 's' cannot be less than 10!";
                return;
            } else
                annealLossValues.resize(static_cast<size_t>(std::round(value)));
            break;
        case 'a':
            if (value > 1.f) {
                std::cout << "Parameter 'a' cannot be more than 1!";
                return;
            }
            break;
        default:
            break;
    }

    auto m = HyperParams.find(param_name);
    if (m != HyperParams.end())
        (*m->second).setCell(0, 0, value);
    else {
        std::cout << "Invalid parameter name: " << param_name << std::endl;
        return;
    }

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

f32 ClassificationNN::GetAccuracy(std::pair<const Matrix2D &, const Matrix2D &> sample) {
    auto[inputs, outputs] = sample;
    auto[Inputs, Outputs] = IO;
    Inputs->assignData(&inputs);
    Outputs->assignData(&outputs);
    ForwardPropagation(AccuracyLayer.second.get());
    AccuracyLayer.first->followProp();
    f32 accuracy = AccuracyLayer.first->getData()(0, 0);
    accuracy /= static_cast<f32>(DataSet.GetBatchSize());
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
    for (auto &Weight : Layers)
        Weight->subGrad(GET_PARAM('l'));
}

void ClassificationNN::Serialize(const char *filename) {
    std::ofstream out(filename, std::ios::binary);
    // u32 layers
    // [for each layer]
    // u32 rows
    // u32 cols
    // f32 [data]
    u32 layers = WeightLayers.size();
    out.write((const char *) &layers, sizeof(u32));
    for (auto&& p : WeightLayers) {
        auto &L = *p;
        u32 r = L.getData().getRows();
        u32 c = L.getData().getCols();

        out.write((const char *) &r, sizeof(u32));
        out.write((const char *) &c, sizeof(u32));
        out.write((const char *) &(L.getData()(0, 0)), r * c * sizeof(f32));
    }
}

void ClassificationNN::Deserialize(const char *) {

}

void ClassificationNN::Use() {

}
