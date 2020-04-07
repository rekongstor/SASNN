#include "../../../../../include/Layer/LayerDecorators/GradientDescent/GradientDescentStochastic.h"

void GradientDescentStochastic::subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) {
    this->learningRate.setCell(0, 0, step);
    weights.EachCellOperator(weights, grad, learningRate, [](const f32 l, const f32 r, const f32 e) -> f32 {
        return l - r * e;
    });
}

GradientDescentStochastic::GradientDescentStochastic() : learningRate(1, 1) {}
