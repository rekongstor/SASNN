#include "../include/Layer/Simple/LayerWeights.h"
#include "../include/Layer/Simple/LayerData.h"
#include "../include/Layer/Functional/LayerStableSoftMax.h"
#include "../include/Layer/Functional/LayerSoftMax.h"
#include "../include/Layer/Functional/LayerCrossEntropyLoss.h"
#include "../include/Layer/Functional/LayerFullyConnected.h"
#include "../include/Layer/Functional/LayerL2Reg.h"
#include "../include/Layer/Accuracy/LayerClassificationAccuracy.h"
#include "../include/Layer/NLF/LayerLeakyReLU.h"
#include "../include/Layer/NLF/LayerReLU.h"
#include "../include/Layer/NLF/LayerTanh.h"
#include "../include/Layer/NLF/LayerSigmoid.h"
#include "../include/Layer/Functional/LayerSum.h"
#include "../include/Layer/Functional/LayerBatchNormalization.h"
#include "../include/Layer/Simple/LayerWeightsDecorators/Initializer/InitializerXavier.h"
#include "../include/Layer/Simple/LayerWeightsDecorators/Initializer/InitializerUniform.h"
#include "../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentStochastic.h"
#include "../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentMomentum.h"
#include "../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentAdaGrad.h"
#include "../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentAdam.h"
#include "../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentRMSProp.h"
#include "../include/Layer/Functional/LayerExp.h"
#include "../include/Layer/Functional/LayerLog.h"

void test() {
    Layer *inp = new LayerWeights(2, 3, new InitializerUniform(0, 1), new GradientDescentStochastic);
    Layer *out = new LayerWeights(2, 4, new InitializerUniform(0, 1), new GradientDescentStochastic);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            out->getData().Fill(0.f);
    out->getData().setCell(0, 0, 1.f);
    out->getData().setCell(1, 1, 1.f);
    Layer *weight = new LayerWeights(3, 4, new InitializerUniform(-1, 1), new GradientDescentStochastic);
    Matrix2D l2r(1.f);
    Layer *l2rd = new LayerData(l2r);
    Layer *l2reg = new LayerL2Reg(*weight, *l2rd, false);
    Layer *bias = new LayerWeights(1, 4, new InitializerUniform(-1, 1), new GradientDescentStochastic);
    Layer *fc = new LayerFullyConnected(*inp, *weight);
    Layer *neurons = new LayerSum(*fc, *bias);
    Layer *norma = new LayerBatchNormalization(*neurons);
    Layer *sm = new LayerStableSoftMax(*norma, true);
    Layer *l = new LayerCrossEntropyLoss(*sm, *out);
    Layer *loss = new LayerSum(*l, *l);
    l2reg->followProp();
    fc->followProp();
    neurons->followProp();
    norma->followProp();
    sm->followProp();
    l->followProp();
    loss->followProp();
    loss->getGrad()->Fill(1.f);
    loss->backProp();
    l->backProp();
    sm->backProp();
    norma->backProp();
    neurons->backProp();
    fc->backProp();
    l2reg->backProp();

    auto &target = weight;
    Matrix2D anal = *target->getGrad();
    // numeric gradient
    f32 delta = 0.001;

    f32 f = loss->getData()(0, 0);
    for (int i = 0; i < target->getGrad()->getRows(); ++i)
        for (int j = 0; j < target->getGrad()->getCols(); ++j) {
            auto &g = target->getData();
            g.setCell(i, j, g(i, j) + delta);
            l2reg->followProp();
            fc->followProp();
            neurons->followProp();
            norma->followProp();
            sm->followProp();
            l->followProp();
            loss->followProp();
            target->getGrad()->setCell(i, j, (loss->getData()(0, 0) - f) / delta);
            g.setCell(i, j, g(i, j) - delta);
        }
    Matrix2D num = *target->getGrad();
    Matrix2D difference(num.getRows(), num.getCols());
    for (int i = 0; i < num.getRows(); ++i)
        for (int j = 0; j < num.getCols(); ++j)
            difference.setCell(i, j, num(i, j) - anal(i, j));
}