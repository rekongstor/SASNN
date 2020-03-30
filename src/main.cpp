#include <iostream>
#include <cmath>
#include "Layer/LayerWeights.h"
#include "Layer/LayerReLU.h"
#include "Layer/LayerData.h"
#include "Layer/LayerFullyConnected.h"
#include "Layer/LayerLeakyReLU.h"
#include "Layer/LayerSigmoid.h"
#include "Layer/LayerTanh.h"
#include "Layer/LayerSoftMaxRow.h"
#include "Layer/LayerSoftMaxCol.h"

int main() {
    Layer *weights = new LayerWeights(2, 4, static_cast<f32>(1));
    Layer *softmax = new LayerSoftMaxCol(*weights);
    softmax->getGrad()->Fill(.001f);
    for (int x = 0; x < 1000; ++x) {
        softmax->followProp();
        softmax->backProp();
        weights->subGrad();
        for (size_t i = 0; i < softmax->getData().getRows(); ++i) {
            for (size_t j = 0; j < softmax->getData().getCols(); ++j)
                std::cout << softmax->getData().operator()(i, j) << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
    return 0;
}
