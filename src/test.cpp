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

void test() {
    Layer *inp = new LayerWeights(2, 3, new InitializerUniform(0, 1), new GradientDescentStochastic);
    Layer *out = new LayerWeights(2, 4, new InitializerUniform(0, 1), new GradientDescentStochastic);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            out->getData().Fill(0.f);
        out->getData().setCell(0,0,1.f);
        out->getData().setCell(1,1,1.f);
    Layer *weight = new LayerWeights(3, 4, new InitializerUniform(0,10), new GradientDescentStochastic);
    Layer *fc = new LayerFullyConnected(*inp, *weight);
    Layer *sm = new LayerSoftMax(*fc, true);
    Layer *l = new LayerCrossEntropyLoss(*sm, *out);
    f32 delta = 0.001;
    fc->followProp();
    sm->followProp();
    l->followProp();
    l->getGrad()->Fill(1.f);
    l->backProp();
    sm->backProp();
    fc->backProp();
    // analytic gradient
    auto &target = weight;
    Matrix2D anal = *target->getGrad();
    weight->clearGrad();
    fc->clearGrad();
    sm->clearGrad();

    f32 f = l->getData()(0, 0);
    for (int i = 0; i < target->getGrad()->getRows(); ++i)
        for (int j = 0; j < target->getGrad()->getCols(); ++j) {
            auto &g = target->getData();
            g.setCell(i, j, g(i, j) + delta);
            fc->followProp();
            sm->followProp();
            l->followProp();
            target->getGrad()->setCell(i, j, (l->getData()(0, 0) - f) / delta);
            g.setCell(i, j, g(i, j) - delta);
        }
    Matrix2D num = *target->getGrad();
    Matrix2D difference(num.getRows(), num.getCols());
    for (int i = 0; i < num.getRows(); ++i)
        for (int j = 0; j < num.getCols(); ++j)
            difference.setCell(i,j,num(i,j) - anal(i,j));

}