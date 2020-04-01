#include "../../../../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentStochastic.h"

void GradientDescentStochastic::subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) {
    this->learningRate.setCell(0, 0, step);
    grad.EachCellOperator(grad, learningRate, [](const f32 l, const f32 r) -> f32 {
        return -l + l * r;
    });
    weights.EachCellOperator(weights, grad, [](const f32 l, const f32 r) -> f32 {
        return l - r;
    });
}

GradientDescentStochastic::GradientDescentStochastic() : learningRate(1, 1) {}
