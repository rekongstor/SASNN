#include <iostream>
#include <cmath>
#include "../include/Layer/LayerWeights.h"


int main() {
    Layer* layerWeights = new LayerWeights(32,32, static_cast<f32>(2));
    delete layerWeights;
    return 0;
}
