#include <iostream>
#include <cmath>
#include "Layer/LayerWeights.h"
#include "Layer/LayerReLU.h"
#include "Layer/LayerData.h"
#include "Layer/LayerFullyConnected.h"


int main() {

    Layer *weights = new LayerWeights(1, 4, static_cast<f32>(2));

    Matrix2D a(4, 2);
    for (size_t i = 0; i < a.getRows(); ++i)
        for (size_t j = 0; j < a.getCols(); ++j)
            a(i,j) = i * a.getCols() + j;
    Layer *data = new LayerData(a);


    Layer *mul = new LayerFullyConnected(*weights, *data);
    mul->getGrad()->Fill(1.f);
    mul->followProp();
    mul->backProp();
    return 0;
}
