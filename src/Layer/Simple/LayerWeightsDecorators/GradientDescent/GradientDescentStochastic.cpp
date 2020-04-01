#include "../../../../../include/Layer/Simple/LayerWeightsDecorators/GradientDescent/GradientDescentStochastic.h"

void GradientDescentStochastic::subGrad(Matrix2D &weights, const Matrix2D &grad) {
    weights.EachCellOperator(weights, grad, [](const f32 l, const f32 r) -> f32 {
        return l - r;
    });
}
