#include <iostream>
#include <cmath>
#include "Layer/LayerWeights.h"
#include "Layer/LayerData.h"
#include "Layer/LayerSumCols.h"

int main() {
    Layer *weights = new LayerWeights(4, 4, static_cast<f32>(1));
    Layer *layer = new LayerSumCols(*weights);
    layer->getGrad()->Fill(0.1f);
    layer->followProp();
    layer->backProp();

    return 0;
}
