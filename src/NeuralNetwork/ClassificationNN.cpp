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
#include "../../include/Layer/Functional/LayerBatchNormalization.h"
#include "../../include/Layer/Simple/LayerWeightsDecorators/Initializer/InitializerXavier.h"
#include "../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentStochastic.h"
#include "../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentMomentum.h"
#include "../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentAdaGrad.h"
#include "../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentAdam.h"
#include "../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentRMSProp.h"


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
    for (int i = 0; i < annealLossValues.size(); ++i) {
        top += (static_cast<f32>(i) - x_avr) * (annealLossValues[i] - y_avr);
        btm += (static_cast<f32>(i) - x_avr) * (static_cast<f32>(i) - x_avr);
    }
    a = top / btm;
    //printf("Front: %f; Back: %f Mean: %f; a:%f\n", annealLossValues.front(), annealLossValues.back(), annMean, a);
    if (annealLossValues.front() < annMean || a > -0.05f) {
        ModifyParam('l', GET_PARAM('l') * GET_PARAM('a'));
        printf("Loss is growing! Learning rate was decreased! Current value: %f\n", GET_PARAM('l'));
    } else if (a < -0.1f) {
        ModifyParam('l', GET_PARAM('l') / GET_PARAM('a'));
        printf("Loss is decreasing! Learning rate was increased! Current value: %f\n", GET_PARAM('l'));
    }
}

ClassificationNN::ClassificationNN(std::vector<u32> &&layers, Dataset &dataset) : DataSet(dataset) {
    // Setting hyper-parameters. Modifiable
    ADD_PARAM('l', 0.0001f); // Learning rate
    ADD_PARAM('r', 40.f); // Regularization
    ADD_PARAM('s', 5.f); // Steps
    ADD_PARAM('a', 0.9f); // Annealing multiplier
    annealLossValues.resize(std::round(GET_PARAM('s')));

    // Setting dataset layers
    auto[train_inputs, train_outputs] = DataSet.GetTrainSample(false);
    auto Input = ADD_LAYER(new LayerData(train_inputs)); // [batch_size x inputs]
    auto Output = ADD_LAYER(new LayerData(train_outputs)); // [batch_size x outputs]
    IO = {Input, Output};

    // Setting hyper-parameter layers
    auto L2RegParam = ADD_LAYER(new LayerData(*HyperParams['r']));
    L2RegularizationLayer = L2RegParam;
    // Setting FullyConnected architecture
    auto Weights = ADD_LAYER(new LayerWeights(dataset.GetInputs(), dataset.GetOutputs(), new InitializerXavier(static_cast<f32>(dataset.GetInputs())), new GradientDescentAdam));

    WeightsLayers.emplace_back(Weights);
    auto FullyConnected = ADD_LAYER(new LayerFullyConnected(*Input, *Weights)); // [batch_size x outputs]
    //auto BatchNormalization = ADD_LAYER(new LayerBatchNormalization(*FullyConnected));

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
    switch (param_name) {
        case 's':
            if (std::round(value) < 10) {
                std::cout << "Parameter 's' cannot be less than 10!";
                return;
            } else
                annealLossValues.resize(std::round(value));
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
    for (auto &Weight : WeightsLayers)
        Weight->subGrad(GET_PARAM('l'));
}

void ClassificationNN::Serialize(const char *filename) {
    std::ofstream out(filename);
    for (auto &W : WeightsLayers)
        for (size_t j = 0; j < W->getData().getCols(); ++j)
            for (size_t i = 0; i < W->getData().getRows(); ++i)
                out << W->getData()(i, j) << std::endl;
}
