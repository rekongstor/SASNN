#include <cmath>
#include "../../../../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentRMSProp.h"

void GradientDescentRMSProp::subGrad(Matrix2D &weights, Matrix2D &grad, f32 step) {
    if (!accumulated)
        accumulated = std::make_shared<Matrix2D>(grad.getRows(), grad.getCols());
    if (!adaptiveLearningRate)
        adaptiveLearningRate = std::make_shared<Matrix2D>(grad.getRows(), grad.getCols());
    learningRate.setCell(0, 0, step);
    
    accumulated->EachCellOperator(*accumulated, grad, [](const f32 l, const f32 r) -> f32 {
        return l + r * r;
    });
    adaptiveLearningRate->EachCellOperator(learningRate, *accumulated, [](const f32 l, const f32 r) -> f32 {
        return l / sqrtf(r);
    });
    grad.EachCellOperator(grad, *adaptiveLearningRate, [](const f32 l, const f32 r) -> f32 {
        return -l + l * r;
    });
    weights.EachCellOperator(weights, grad, [](const f32 l, const f32 r) -> f32 {
        return l - r;
    });
}

GradientDescentRMSProp::GradientDescentRMSProp(f32 rho) : learningRate(1, 1), rho(rho) {}
