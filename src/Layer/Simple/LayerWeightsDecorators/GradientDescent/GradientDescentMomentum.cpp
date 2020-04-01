#include "../../../../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentMomentum.h"

void GradientDescentMomentum::subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) {
    if (!velocity)
        velocity = std::make_shared<Matrix2D>(grad.getRows(), grad.getCols());

    learningRate.setCell(0, 0, step / momentum(0, 0));

    velocity->EachCellOperator(*velocity, grad, learningRate, [](const f32 l, const f32 r, const f32 e) -> f32 {
        return l - r * e;
    }, &momentum);
    weights.EachCellOperator(weights, *velocity, [](const f32 l, const f32 r) -> f32 {
        return l + r;
    });
}

GradientDescentMomentum::GradientDescentMomentum(f32 velocity) : learningRate(1, 1),
                                                                 momentum(velocity) {}

