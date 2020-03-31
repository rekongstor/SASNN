#include <iostream>
#include "../include/Layer/Simple/LayerWeights.h"
#include "../include/Layer/Simple/LayerData.h"
#include "Layer/Functional/LayerStableSoftMax.h"
#include "Layer/Functional/LayerLog.h"
#include "Layer/Functional/LayerGT.h"
#include "Layer/Functional/LayerSumAll.h"


int main() {
    Layer *Weights = new LayerWeights(2, 10);
    Layer *SoftMax = new LayerStableSoftMax(*Weights);
    Layer *Log = new LayerLog(*SoftMax);

    Matrix2D gt(2, 10);
    gt.Fill(-.1f);
    gt.setCell(0, 5, 1.f);
    gt.setCell(1, 2, 1.f);
    Layer *L = new LayerData(gt);

    Layer *GT = new LayerGT(*Log, *L);
    Layer *Loss = new LayerSumAll(*GT);
    Loss->getGrad()->setCell(0, 0, -.1f);
for (int i = 0; i < 1000; ++i) {
    SoftMax->followProp();
    Log->followProp();
    GT->followProp();
    Loss->followProp();

    Weights->clearGrad();
    SoftMax->clearGrad();
    Log->clearGrad();
    GT->clearGrad();

    Loss->backProp();
    GT->backProp();
    Log->backProp();
    SoftMax->backProp();

    SoftMax->getData().Print();
    Weights->subGrad();
}
    return 0;
}
