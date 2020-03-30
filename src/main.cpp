#include <iostream>
#include <cmath>
#include "Layer/LayerWeights.h"
#include "Layer/LayerReLU.h"
#include "Layer/LayerData.h"
#include "Layer/LayerFullyConnected.h"
#include "Layer/LayerLeakyReLU.h"
#include "Layer/LayerSigmoid.h"
#include "Layer/LayerTanh.h"

int main() {
    Layer *weights = new LayerWeights(2, 4, static_cast<f32>(1));
    Layer *nlf = new LayerTanh(*weights);
    nlf->getGrad()->Fill(1.f);
    nlf->followProp();
    nlf->backProp();
    return 0;
}
