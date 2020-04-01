#include "../../../../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentMomentum.h"

void GradientDescentMomentum::subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) {
    learningRate.setCell(0, 0, step);
    if (!velocity)
        velocity = std::make_shared<Matrix2D>(grad.getRows(), grad.getCols());

    velocity->EachCellOperator(*velocity, grad, [](const f32 l, const f32 r) -> f32 {
        return l - r;
    }, &momentum);
    weights.EachCellOperator(weights, *velocity, [](const f32 l, const f32 r) -> f32 {
        return l + r;
    });
}

GradientDescentMomentum::GradientDescentMomentum(f32 velocity) : momentum(velocity),
                                                                 learningRate(1, 1) {}

