#include <cmath>
#include "../../../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentRMSProp.h"

void GradientDescentRMSProp::subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) {
    if (!accumulated)
        accumulated = std::make_shared<Matrix2D>(grad.getRows(), grad.getCols());
    learningRate.setCell(0, 0, step);

    accumulated->EachCellOperator(*accumulated, grad, rho, [](const f32 l, const f32 r, const f32 e) -> f32 {
        return l * e + (1-e) * (r * r);
    });

    weights.EachCellOperator(weights, grad, learningRate, *accumulated, [](const f32 w, const f32 g, const f32 l, const f32 a) -> f32 {
        return w - l / sqrtf(a) * g;
    });
}

GradientDescentRMSProp::GradientDescentRMSProp(f32 rho) : learningRate(1, 1), rho(rho) {}
