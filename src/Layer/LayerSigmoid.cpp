#include "LayerSigmoid.h"
#include <cmath>

void LayerSigmoid::followProp() {
    data.EachCellOperator(left.getData(), [](const f32 left) -> f32 {
        return 1.f / (1.f + expf(-left));
    });
}

void LayerSigmoid::backProp() {
    if (left.getGrad() != nullptr) {
        Matrix2D &g = *left.getGrad();
        g.EachCellOperator(data, [](const f32 left) -> f32 {
            return left * (1 - left);
        }, &grad);
    }
}

LayerSigmoid::LayerSigmoid(Layer &left) : LayerDynamic(left.getData().getRows(), left.getData().getCols()),
                                          left(left) {}
