#include <iostream>
#include <cmath>
#include "../include/Layer/LayerWeights.h"
#include "Layer/LayerReLU.h"


int main() {
    Layer* layerWeights = new LayerWeights(8,4, static_cast<f32>(2));
    Layer* pReLu = new LayerReLU(*layerWeights);
    pReLu->followProp();
    // imitate initial 1.f gradient
    auto& g = *pReLu->getGrad();
    for (size_t i = 0; i < g.getRows(); ++i)
        for (size_t j = 0; j < g.getCols(); ++j)
            g(i,j) = 1.f;
    pReLu->backProp();
    delete layerWeights;
    return 0;
}
