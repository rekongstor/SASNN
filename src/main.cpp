#include <iostream>
#include "../include/Layer/Simple/LayerWeights.h"
#include "../include/Layer/Simple/LayerData.h"
#include "Layer/Functional/LayerStableSoftMax.h"


int main() {
    Layer *Weights = new LayerWeights(2, 2);
    auto l = Weights->getGrad();
    for (size_t i = 0; i < l->getRows(); ++i)
        l->setCell(i, i, -1.f);
    Weights->subGrad();

    Layer *SoftMax = new LayerStableSoftMax(*Weights);
    auto g = SoftMax->getGrad();
    g->Fill(0.f);
    for (size_t i = 0; i < g->getRows(); ++i)
        g->setCell(i, i, 1.f);

    SoftMax->followProp();
    SoftMax->backProp();
    Weights->subGrad();

    return 0;
}
