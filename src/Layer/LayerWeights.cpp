#include "LayerWeights.h"

void LayerWeights::followProp() {

}

void LayerWeights::backProp() {

}

LayerWeights::LayerWeights(size_t rows, size_t cols, bool random) : LayerDynamic(rows, cols) {
    if (random) {
        RandomUniform randomUniform(-.5f, .5f);
        Random *rng = &randomUniform;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                data(i, j) = rng->Next();
    } else
        data.Clean();
}

LayerWeights::LayerWeights(size_t rows, size_t cols, f32 xavier_inputs) : LayerDynamic(rows, cols) {
    RandomGaussian randomGaussian = RandomGaussian(
            0.f,
            sqrtf(2.f / xavier_inputs));
    Random *rng = &randomGaussian;

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data(i, j) = rng->Next();
}
