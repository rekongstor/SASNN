
#include "../../../include/Layer/LayerDecorators/DecoratorInitializer.h"
#include "../../../include/Layer/LayerDecorators/DecoratorGradientDescent.h"
#include "../../../include/Layer/Simple/LayerWeights.h"

void LayerWeights::followProp() {

}

void LayerWeights::backProp() {

}

LayerWeights::LayerWeights(size_t rows, size_t cols, DecoratorInitializer *decoratorInitializer, DecoratorGradientDescent *decoratorGradientDescent) :
        LayerDynamic(rows, cols),
        gradientDescent(decoratorGradientDescent),
        initializer(decoratorInitializer) {
    initializer->Initialize(data);
}

void LayerWeights::subGrad(f32 step) {
    gradientDescent->subGrad(data, grad, step);
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

