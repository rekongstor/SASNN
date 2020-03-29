#include <iostream>
#include <cmath>
#include "Layer/LayerWeights.h"
#include "Layer/LayerReLU.h"
#include "Layer/LayerData.h"
#include "Layer/LayerFullyConnected.h"
#include "Layer/LayerLeakyReLU.h"

int main() {

    Layer *weights = new LayerWeights(2, 4, static_cast<f32>(2));

    Matrix2D d(0.01f);
    Layer *leak = new LayerData(d);
    Layer *pLeakyReLu = new LayerLeakyReLU(*weights, *leak);
    pLeakyReLu->getGrad()->Fill((1.f));
    pLeakyReLu->followProp();
    pLeakyReLu->backProp();

    return 0;
}
