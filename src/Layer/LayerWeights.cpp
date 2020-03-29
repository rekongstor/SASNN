#include "../../include/Layer/LayerWeights.h"

void LayerWeights::followProp() {

}

void LayerWeights::backProp(Matrix2D &grad) {

}

void LayerWeights::clearGrad() {
    grad.Clean();
}

LayerWeights::LayerWeights(size_t rows, size_t cols, bool random) : data(rows, cols),
                                                                    grad(rows, cols, true) {
    if (random) {
        RandomUniform randomUniform;
        Random *rng = &randomUniform;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                data(i, j) = rng->Next();
    } else
        data.Clean();
}

LayerWeights::LayerWeights(size_t rows, size_t cols, size_t xavier_inputs) : data(rows, cols),
                                                                             grad(rows, cols, true) {
    RandomGaussian randomGaussian = RandomGaussian(
            0.f,
            sqrtf(2.f / static_cast<f32>(xavier_inputs)));
    Random *rng = &randomGaussian;

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data(i, j) = rng->Next();
}
