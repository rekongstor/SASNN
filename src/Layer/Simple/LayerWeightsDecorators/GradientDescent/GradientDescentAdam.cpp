#include <cmath>
#include "../../../../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentAdam.h"

void GradientDescentAdam::subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) {
    if (!velocity)
        velocity = std::make_shared<Matrix2D>(grad.getRows(), grad.getCols());
    if (!accumulated)
        accumulated = std::make_shared<Matrix2D>(grad.getRows(), grad.getCols());

    velocity->EachCellOperator(*velocity, grad, momentum, [](const f32 v, const f32 g, const f32 m) -> f32 {
        return m * v + (1.f - m) * g;
    });
    accumulated->EachCellOperator(*accumulated, grad, rho, [](const f32 a, const f32 g, const f32 r) -> f32 {
        return r * a + (1.f - r) * g * g;
    });
    weights.EachCellOperator(weights, *velocity, learningRate, *accumulated, [](const f32 w, const f32 v, const f32 l, const f32 a) -> f32 {
        return w - l / sqrt(a) * v;
    });
    learningRate.setCell(0, 0, step);
}


GradientDescentAdam::GradientDescentAdam(f32 momentum, f32 rho) : learningRate(1, 1),
                                                                  momentum(momentum),
                                                                  rho(rho) {}
