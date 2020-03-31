#include <iostream>
#include "../include/Layer/Simple/LayerWeights.h"
#include "../include/Layer/Simple/LayerData.h"
#include "Layer/Functional/LayerStableSoftMax.h"
#include "Layer/Functional/LayerCrossEntropyLoss.h"


int main() {
    Matrix2D gt(2, 10);
    gt.Fill(-.1f);
    gt.setCell(0, 5, 1.f);
    gt.setCell(1, 2, 1.f);
    Layer *L = new LayerData(gt);

    Layer *Weights = new LayerWeights(2, 10);
    Layer *SoftMax = new LayerStableSoftMax(*Weights);


    Layer *CEL = new LayerCrossEntropyLoss(*SoftMax, *L);
    CEL->getGrad()->setCell(0, 0, .1f);
    for (int i = 0; i < 9; ++i) {
        SoftMax->followProp();
        CEL->followProp();

        CEL->backProp();
        SoftMax->backProp();

        SoftMax->getData().Print();
        Weights->subGrad();
    }
    return 0;
}
