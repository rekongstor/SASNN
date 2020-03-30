#include <iostream>
#include <cmath>
#include "Layer/LayerWeights.h"
#include "Layer/LayerReLU.h"
#include "Layer/LayerData.h"
#include "Layer/LayerFullyConnected.h"
#include "Layer/LayerLeakyReLU.h"
#include "Layer/LayerSigmoid.h"

int main() {
    Matrix2D leak(0.01f);
    Layer *lr = new LayerData(leak);
    Layer *weights = new LayerWeights(2, 4, static_cast<f32>(1));
    Layer *nlf = new LayerReLU(*weights);
    nlf->getGrad()->Fill(1.f);
    nlf->followProp();
    nlf->backProp();
    weights->clearGrad();
    nlf = new LayerLeakyReLU(*weights, *lr);
    nlf->getGrad()->Fill(1.f);
    nlf->followProp();
    nlf->backProp();
    weights->clearGrad();
    nlf = new LayerSigmoid(*weights);
    nlf->getGrad()->Fill(1.f);
    nlf->followProp();
    nlf->backProp();
    return 0;
}
