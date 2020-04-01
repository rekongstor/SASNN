#include "../../../include/Layer/Simple/LayerWeights.h"
#include "../../../include/Layer/Simple/LayerWeightsDecorators/DecoratorInitializer.h"
#include "../../../include/Layer/Simple/LayerWeightsDecorators/DecoratorGradientDescent.h"

void LayerWeights::followProp() {

}

void LayerWeights::backProp() {

}

LayerWeights::LayerWeights(size_t rows, size_t cols, DecoratorInitializer *decoratorInitializer, DecoratorGradientDescent *decoratorGradientDescent) :
        LayerDynamic(rows, cols),
        gradLength(1, 1),
        gradientDescent(decoratorGradientDescent),
        initializer(decoratorInitializer) {
    initializer->Initialize(data);
}

void LayerWeights::subGrad(f32 step) {
    // Gradient normalization
    gradLength.MergeCellsOperator(grad, [](const f32 l, const f32 r) -> f32 {
        return l + r * r;
    }, nullptr, [](const f32 l) -> f32 {
        return l * l;
    });
    gradLength.setCell(0, 0, sqrtf(gradLength(0, 0)) / step);
    grad.CellOperator(grad, gradLength, [](const f32 l, const f32 r) -> f32 {
        return -l + l / r;
    });

    // Gradient Descent
    gradientDescent->subGrad(data, grad);
}

void LayerWeights::assignData(const Matrix2D *d) {
    data = *d;
}

void LayerWeights::clearGrad() {
    grad.EachCellOperator(grad, [](const f32 l) -> f32 {
        return l * 0.9f;
    });
    grad.Clean();
}

