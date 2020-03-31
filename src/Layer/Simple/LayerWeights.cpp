#include "../../../include/Layer/Simple/LayerWeights.h"

void LayerWeights::followProp() {

}

void LayerWeights::backProp() {

}

LayerWeights::LayerWeights(size_t rows, size_t cols, bool random) : LayerDynamic(rows, cols),
                                                                    gradLength(1, 1) {
    if (random) {
        RandomUniform randomUniform(-.5f, .5f);
        Random *rng = &randomUniform;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                data.setCell(i, j, rng->Next());
    } else
        data.Clean();
}

LayerWeights::LayerWeights(size_t rows, size_t cols, f32 xavier_inputs) : LayerDynamic(rows, cols),
                                                                          gradLength(1, 1) {
    RandomGaussian randomGaussian = RandomGaussian(
            0.f,
            sqrtf(2.f / xavier_inputs));
    Random *rng = &randomGaussian;

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data.setCell(i, j, rng->Next());
}

void LayerWeights::subGrad() {
    // Gradient normalization
    gradLength.MergeCellsOperator(grad, [](const f32 l, const f32 r) -> f32 {
        return l + r * r;
    }, nullptr, [](const f32 l) -> f32 {
        return l * l;
    });
    gradLength.setCell(0, 0, sqrtf(gradLength(0, 0)));
    grad.CellOperator(grad, gradLength, [](const f32 l, const f32 r) -> f32 {
        return l / r;
    });

    data.EachCellOperator(data, grad, [](const f32 l, const f32 r) -> f32 {
        return l - r;
    });
}

void LayerWeights::assignData(const Matrix2D *d) {
    data = *d;
}

